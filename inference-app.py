import os
import torch
import gradio as gr
from transformers import AutoModel, AutoTokenizer
import time
import random
import numpy as np

# Import your model classes
from dual_mode_model_training import (
    EnhancedFusionLayer,
    DualDecoderModel,
    create_dual_decoder_model
)

class DualModelInferenceApp:
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.joint_mode = True
        
        print("Initializing dual model inference app...")
        self.setup_models_and_tokenizers()
        
    def setup_models_and_tokenizers(self):
        """Load models, tokenizers, and set up the dual decoder model"""
        print("Loading tokenizers...")
        self.tokenizer1 = AutoTokenizer.from_pretrained(self.config["model_dir1"])
        self.tokenizer2 = AutoTokenizer.from_pretrained(self.config["model_dir2"])
        
        # Set up model with inverted GPU allocation
        print("Setting up models with inverted GPU allocation...")
        self.dual_model = self.load_dual_model()
        
        # Set up device map for generation
        self.device_map = {
            'model1': 'cuda:1',
            'model2': 'cuda:0',
            'fusion': 'cuda:0'
        }
        
        # Move model components to their devices for inference
        self.dual_model.model1.to(self.device_map['model1'])
        self.dual_model.model2.to(self.device_map['model2'])
        self.dual_model.fusion_layer.to(self.device_map['fusion'])
        self.dual_model.lm_head1.to(self.device_map['fusion'])
        self.dual_model.lm_head2.to(self.device_map['fusion'])
        
        print("Models loaded and placed on devices:")
        print(f"Model1 (Qwen) on {next(self.dual_model.model1.parameters()).device}")
        print(f"Model2 (Llama) on {next(self.dual_model.model2.parameters()).device}")
        print(f"Fusion on {next(self.dual_model.fusion_layer.parameters()).device}")
        
        # Set to evaluation mode
        self.dual_model.eval()
    
    
    def load_dual_model(self):
        """Load the dual decoder model and trained weights"""
        print("Loading base models...")
        gpu_count = torch.cuda.device_count()
        
        # Define device map
        device_map = {
            'model1': 'cuda:1',  # First model on GPU 1
            'model2': 'cuda:0',  # Second model on GPU 0
            'fusion': 'cuda:0'   # Fusion layer on GPU 0
        }
        
        # Load base models with explicit device placement
        print(f"Loading Model1 onto {device_map['model1']}...")
        torch.cuda.set_device(int(device_map['model1'].split(':')[1]))
        model1 = AutoModel.from_pretrained(self.config["model_dir1"])
        model1.to(device_map['model1'])
        
        print(f"Loading Model2 onto {device_map['model2']}...")
        torch.cuda.set_device(int(device_map['model2'].split(':')[1]))
        model2 = AutoModel.from_pretrained(self.config["model_dir2"])
        model2.to(device_map['model2'])
        
        # Create dual model
        print("Creating dual decoder model...")
        dual_model = DualDecoderModel(
            model1,
            model2,
            self.tokenizer1,
            self.tokenizer2,
            self.config["fusion_output_dim"],
            freeze_base_models=True,
            device_map=device_map
        )
        
        # Check if checkpoint exists and load it
        if os.path.exists(self.checkpoint_path):
            print(f"Loading checkpoint from {self.checkpoint_path}")
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'fusion_layer' in checkpoint:
                    state_dict = checkpoint
                elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    print("Warning: Unexpected checkpoint format, trying to load directly...")
                    state_dict = checkpoint
                
                if 'fusion_layer' in state_dict:
                    dual_model.fusion_layer.load_state_dict(state_dict['fusion_layer'])
                if 'proj1' in state_dict:
                    dual_model.proj1.load_state_dict(state_dict['proj1'])
                if 'proj2' in state_dict:
                    dual_model.proj2.load_state_dict(state_dict['proj2'])
                if 'fused_lm_head' in state_dict:
                    dual_model.fused_lm_head.load_state_dict(state_dict['fused_lm_head'])
                if 'lm_head1' in state_dict:
                    dual_model.lm_head1.load_state_dict(state_dict['lm_head1'])
                if 'lm_head2' in state_dict:
                    dual_model.lm_head2.load_state_dict(state_dict['lm_head2'])
                if 'task_classifier' in state_dict:
                    dual_model.task_classifier.load_state_dict(state_dict['task_classifier'])
                print("Checkpoint loaded successfully")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Continuing with initialized model")
        else:
            print(f"No checkpoint found at {self.checkpoint_path}, using initialized model")
        
        return dual_model
        
    def toggle_joint_mode(self):
        """Toggle between joint and independent input modes"""
        self.joint_mode = not self.joint_mode
        return "Joint Mode: ON" if self.joint_mode else "Joint Mode: OFF"
    
    def sync_inputs(self, text, other_text, source):
        """Synchronize inputs when in joint mode"""
        if self.joint_mode:
            return text, text
        else:
            return text, other_text
    
    def generate_dual_streaming(self, input_ids1, input_ids2, tokenizer1, tokenizer2, 
                                max_length=100, temperature=0.7, do_sample=True, 
                                attention_mask1=None, attention_mask2=None,
                                return_attention_maps=False, seed=None, feed_mode="default"):
        """Generation function that yields intermediate outputs for streaming with seed and feed mode control."""
        self.dual_model.eval()
        
        # Set initial seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            print(f"Using seed: {seed} for generation")
        else:
            random_seed = random.randint(0, 999999)
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
            print(f"Using random seed: {random_seed} for generation")
            
        # Reset model state to ensure clean generation
        if hasattr(self.dual_model.model1, "reset_cache"):
            self.dual_model.model1.reset_cache()
        if hasattr(self.dual_model.model2, "reset_cache"):
            self.dual_model.model2.reset_cache()
            
        prompt1 = tokenizer1.decode(input_ids1[0], skip_special_tokens=True)
        prompt2 = tokenizer2.decode(input_ids2[0], skip_special_tokens=True)
        current_text1 = prompt1
        current_text2 = prompt2
        
        current_ids1 = input_ids1.to(self.device_map['model1'])
        current_ids2 = input_ids2.to(self.device_map['model2'])
        if len(current_ids1.shape) == 1:
            current_ids1 = current_ids1.unsqueeze(0)
        if len(current_ids2.shape) == 1:
            current_ids2 = current_ids2.unsqueeze(0)
        if attention_mask1 is None:
            attention_mask1 = torch.ones_like(current_ids1).to(self.device_map['model1'])
        else:
            attention_mask1 = attention_mask1.to(self.device_map['model1'])
        if attention_mask2 is None:
            attention_mask2 = torch.ones_like(current_ids2).to(self.device_map['model2'])
        else:
            attention_mask2 = attention_mask2.to(self.device_map['model2'])
        
        inputs_match = False
        if current_ids1.size() == current_ids2.size():
            if torch.equal(current_ids1.cpu(), current_ids2.cpu()):
                inputs_match = True
        mode = "single" if inputs_match else "multi"

        # Ensure gate projection layers exist
        if not hasattr(self.dual_model, 'gate1_proj'):
            print("Creating gate projection layer for model 1")
            self.dual_model.gate1_proj = torch.nn.Linear(
                self.dual_model.fusion_layer.fusion_output_dim, 
                self.dual_model.vocab_size1
            ).to(self.device_map['fusion'])
        
        if not hasattr(self.dual_model, 'gate2_proj'):
            print("Creating gate projection layer for model 2")
            self.dual_model.gate2_proj = torch.nn.Linear(
                self.dual_model.fusion_layer.fusion_output_dim, 
                self.dual_model.vocab_size2
            ).to(self.device_map['fusion'])
        
        start_time = time.time()
        for step in range(max_length):
            # Set per-step seed if provided for reproducibility
            if seed is not None:
                rng_state = torch.get_rng_state()
                np_state = np.random.get_state()
                py_state = random.getstate()
                if torch.cuda.is_available():
                    cuda_rng_state = torch.cuda.get_rng_state_all()
                step_seed = seed + step
                random.seed(step_seed)
                np.random.seed(step_seed)
                torch.manual_seed(step_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(step_seed)
            
            with torch.no_grad():
                try:
                    if step > 0 and (attention_mask1.size(1) != current_ids1.size(1) or 
                                     attention_mask2.size(1) != current_ids2.size(1)):
                        print(f"Warning: Attention mask mismatch at step {step}")
                        attention_mask1 = torch.ones_like(current_ids1, device=self.device_map['model1'])
                        attention_mask2 = torch.ones_like(current_ids2, device=self.device_map['model2'])
                    
                    max_id1 = self.tokenizer1.vocab_size - 1
                    max_id2 = self.tokenizer2.vocab_size - 1
                    
                    if torch.any(current_ids1 > max_id1):
                        print("Warning: Invalid token IDs for model1, fixing...")
                        current_ids1 = torch.clamp(current_ids1, max=max_id1)
                    
                    if torch.any(current_ids2 > max_id2):
                        print("Warning: Invalid token IDs for model2, fixing...")
                        current_ids2 = torch.clamp(current_ids2, max=max_id2)
                    
                    outputs1 = self.dual_model.model1(input_ids=current_ids1, attention_mask=attention_mask1)
                    outputs2 = self.dual_model.model2(input_ids=current_ids2, attention_mask=attention_mask2)
                    
                    hidden1 = outputs1.last_hidden_state if hasattr(outputs1, "last_hidden_state") else outputs1[0]
                    hidden2 = outputs2.last_hidden_state if hasattr(outputs2, "last_hidden_state") else outputs2[0]
                except RuntimeError as e:
                    print(f"Error during model forward pass: {e}")
                    print("Attempting recovery...")
                    if current_ids1.size(1) > 10 and current_ids2.size(1) > 10:
                        recovery_length = min(10, current_ids1.size(1) - 1)
                        print(f"Truncating context to last {recovery_length} tokens")
                        prompt_len1 = len(tokenizer1.encode(prompt1))
                        prompt_len2 = len(tokenizer2.encode(prompt2))
                        if prompt_len1 < current_ids1.size(1) - recovery_length:
                            current_ids1 = torch.cat([
                                current_ids1[:, :prompt_len1],
                                current_ids1[:, -recovery_length:]
                            ], dim=1)
                            attention_mask1 = torch.ones_like(current_ids1, device=self.device_map['model1'])
                        if prompt_len2 < current_ids2.size(1) - recovery_length:
                            current_ids2 = torch.cat([
                                current_ids2[:, :prompt_len2],
                                current_ids2[:, -recovery_length:]
                            ], dim=1)
                            attention_mask2 = torch.ones_like(current_ids2, device=self.device_map['model2'])
                        try:
                            outputs1 = self.dual_model.model1(input_ids=current_ids1, attention_mask=attention_mask1)
                            outputs2 = self.dual_model.model2(input_ids=current_ids2, attention_mask=attention_mask2)
                            
                            hidden1 = outputs1.last_hidden_state if hasattr(outputs1, "last_hidden_state") else outputs1[0]
                            hidden2 = outputs2.last_hidden_state if hasattr(outputs2, "last_hidden_state") else outputs2[0]
                            
                            print("Recovery successful!")
                            current_text1 = tokenizer1.decode(current_ids1[0], skip_special_tokens=True)
                            current_text2 = tokenizer2.decode(current_ids2[0], skip_special_tokens=True)
                            
                            yield current_text1, current_text2, f"Recovered from error at step {step}"
                        except Exception as recovery_error:
                            print(f"Recovery failed: {recovery_error}")
                            yield current_text1, current_text2, f"Generation failed at step {step}: {str(e)}"
                            return
                
                hidden1 = hidden1.to(self.device_map['fusion'])
                hidden2 = hidden2.to(self.device_map['fusion'])
                
                seq_len1 = hidden1.size(1)
                seq_len2 = hidden2.size(1)
                if seq_len1 != seq_len2:
                    max_len = max(seq_len1, seq_len2)
                    if seq_len1 < max_len:
                        padding = torch.zeros(hidden1.size(0), max_len - seq_len1, hidden1.size(2), device=self.device_map['fusion'])
                        hidden1 = torch.cat([hidden1, padding], dim=1)
                        mask_padding = torch.zeros(attention_mask1.size(0), max_len - seq_len1, device=self.device_map['fusion'])
                        attention_mask1_fusion = torch.cat([attention_mask1.to(self.device_map['fusion']), mask_padding], dim=1)
                    else:
                        attention_mask1_fusion = attention_mask1.to(self.device_map['fusion'])
                    if seq_len2 < max_len:
                        padding = torch.zeros(hidden2.size(0), max_len - seq_len2, hidden2.size(2), device=self.device_map['fusion'])
                        hidden2 = torch.cat([hidden2, padding], dim=1)
                        mask_padding = torch.zeros(attention_mask2.size(0), max_len - seq_len2, device=self.device_map['fusion'])
                        attention_mask2_fusion = torch.cat([attention_mask2.to(self.device_map['fusion']), mask_padding], dim=1)
                    else:
                        attention_mask2_fusion = attention_mask2.to(self.device_map['fusion'])
                else:
                    attention_mask1_fusion = attention_mask1.to(self.device_map['fusion'])
                    attention_mask2_fusion = attention_mask2.to(self.device_map['fusion'])
                
                fusion_outputs = self.dual_model.fusion_layer(
                    hidden1, 
                    hidden2, 
                    mask1=attention_mask1_fusion,
                    mask2=attention_mask2_fusion,
                    mode=mode
                )
                
                fused = fusion_outputs["fused"]
                out1 = fusion_outputs["out1"]
                out2 = fusion_outputs["out2"]
                gate1 = fusion_outputs["gate1"]
                gate2 = fusion_outputs["gate2"]
                
                fused_logits = self.dual_model.fused_lm_head(fused)
                logits1 = self.dual_model.lm_head1(out1)
                logits2 = self.dual_model.lm_head2(out2)
                
                # Ensure correct dimensions first (fix the dimension issue):
                fused_logits_last = fused_logits[:, -1, :] if len(fused_logits.shape) > 2 else fused_logits
                logits1_last = logits1[:, -1, :] if len(logits1.shape) > 2 else logits1
                logits2_last = logits2[:, -1, :] if len(logits2.shape) > 2 else logits2

                # Get Llama's vocab size
                vocab_size2 = logits2_last.size(-1)  # Should be 128256

                # Trim the fused_logits to match Llama's vocabulary size
                fused_logits_for_model2 = fused_logits_last[:, :vocab_size2]

                # Add diagnostic log to verify trimming
                if step == 0:  # Only print for first step
                    print(f"Trimmed fused_logits for model2: {fused_logits_for_model2.shape}")

                # Now apply the weighting with the trimmed tensor
                if mode == "single":
                    # Model1 weighting remains the same
                    weighted_logits1 = (1.0 * fused_logits_last + 0.3 * logits1_last) / 1.3
                    
                    # Model2 now gets fused representation access
                    weighted_logits2 = (0.5 * fused_logits_for_model2 + 0.5 * logits2_last) / 1.0
                else:
                    # Multi-task mode
                    weighted_logits1 = (0.2 * fused_logits_last + 0.4 * logits1_last) / 0.6
                    weighted_logits2 = (0.3 * fused_logits_for_model2 + 0.7 * logits2_last) / 1.0
                
                if step == 0:
                    print(f"Last token logits shapes:")
                    print(f"  fused_logits_last: {fused_logits_last.shape}")
                    print(f"  logits1_last: {logits1_last.shape}")
                    print(f"  logits2_last: {logits2_last.shape}")
                
                if len(logits1_last.shape) > 2:
                    logits1_last = logits1_last.squeeze(1)
                if len(logits2_last.shape) > 2:
                    logits2_last = logits2_last.squeeze(1)
                
                if mode == "single":
                    weighted_logits1 = (1.0 * fused_logits_last + 0.3 * logits1_last) / 1.3
                    weighted_logits2 = logits2_last
                else:
                    weighted_logits1 = (0.2 * fused_logits_last + 0.4 * logits1_last) / 0.6
                    weighted_logits2 = logits2_last
                
                if len(weighted_logits1.shape) > 2:
                    weighted_logits1 = weighted_logits1.squeeze(1)
                if len(weighted_logits2.shape) > 2:
                    weighted_logits2 = weighted_logits2.squeeze(1)
                
                if temperature > 0:
                    weighted_logits1 = weighted_logits1 / temperature
                    weighted_logits2 = weighted_logits2 / temperature
                
                if step == 0:
                    print(f"Mode: {mode}")
                    print(f"Vocab sizes - fused: {fused_logits_last.size(-1)}, logits1: {logits1_last.size(-1)}, logits2: {logits2_last.size(-1)}")
                    print(f"Using normalized weights for model1 - {'1.0 fused + 0.3 model1' if mode == 'single' else '0.2 fused + 0.4 model1'}")
                
                if do_sample:
                    if step == 0:
                        print(f"Logits shapes before softmax - model1: {weighted_logits1.shape}, model2: {weighted_logits2.shape}")
                    weighted_logits1 = torch.nan_to_num(weighted_logits1)
                    weighted_logits2 = torch.nan_to_num(weighted_logits2)
                    
                    if len(weighted_logits1.shape) == 1:
                        weighted_logits1 = weighted_logits1.unsqueeze(0)
                    if len(weighted_logits2.shape) == 1:
                        weighted_logits2 = weighted_logits2.unsqueeze(0)
                    
                    probs1 = torch.softmax(weighted_logits1, dim=-1)
                    probs2 = torch.softmax(weighted_logits2, dim=-1)
                    
                    if step == 0:
                        print(f"Probs shapes - model1: {probs1.shape}, model2: {probs2.shape}")
                        print(f"Probs1 sum: {probs1.sum().item()}, has NaN: {torch.isnan(probs1).any().item()}")
                    
                    if len(probs1.shape) > 2:
                        probs1 = probs1[:, -1, :]
                        print(f"Fixed probs1 shape to: {probs1.shape}")
                    if len(probs2.shape) > 2:
                        probs2 = probs2[:, -1, :]
                        print(f"Fixed probs2 shape to: {probs2.shape}")
                    
                    try:
                        next_token1 = torch.multinomial(probs1, num_samples=1)
                        next_token2 = torch.multinomial(probs2, num_samples=1)
                    except RuntimeError as e:
                        print(f"Error during sampling: {e}")
                        print(f"probs1 min: {probs1.min().item()}, max: {probs1.max().item()}")
                        print(f"probs2 min: {probs2.min().item()}, max: {probs2.max().item()}")
                        print("Falling back to greedy decoding")
                        next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                        next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
                else:
                    next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                    next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
                
                if seed is not None:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    torch.set_rng_state(rng_state)
                    if torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(cuda_rng_state)
                
                if len(next_token1.shape) > 2:
                    next_token1 = next_token1.view(next_token1.size(0), -1)
                if len(next_token2.shape) > 2:
                    next_token2 = next_token2.view(next_token2.size(0), -1)
                
                next_token1 = next_token1.to(self.device_map['model1'])
                next_token2 = next_token2.to(self.device_map['model2'])
                
                # Decode tokens for display
                token_text1 = tokenizer1.decode(next_token1[0], skip_special_tokens=True)
                token_text2 = tokenizer2.decode(next_token2[0], skip_special_tokens=True)
                
                if not token_text2 or token_text2.isspace() or token_text2.startswith("<"):
                    token_text2 = tokenizer2.decode(next_token2[0], skip_special_tokens=False)
                    if not token_text2 or token_text2.isspace():
                        token_id = next_token2[0].item()
                        if token_id < len(tokenizer2.vocab):
                            token_text2 = tokenizer2.convert_ids_to_tokens(token_id)
                            print(f"Fallback token2 text: '{token_text2}'")
                
                # Update texts and input sequences based on the selected feed_mode.
                # When feeding tokens from one model to the other, we move the tensor to the appropriate device.
                if feed_mode == "default":
                    current_text1 += token_text1
                    current_text2 += token_text2
                    current_ids1 = torch.cat([current_ids1, next_token1], dim=1)
                    current_ids2 = torch.cat([current_ids2, next_token2], dim=1)
                elif feed_mode == "model1_both":
                    current_text1 += token_text1
                    current_text2 += token_text1
                    # current_ids1 is on device_map['model1'] (already next_token1)
                    current_ids1 = torch.cat([current_ids1, next_token1], dim=1)
                    # For current_ids2 (on device_map['model2']), move next_token1 to that device
                    current_ids2 = torch.cat([current_ids2, next_token1.to(self.device_map['model2'])], dim=1)
                elif feed_mode == "model2_both":
                    current_text1 += token_text2
                    current_text2 += token_text2
                    # current_ids2 is on device_map['model2'] (already next_token2)
                    current_ids2 = torch.cat([current_ids2, next_token2], dim=1)
                    # For current_ids1 (on device_map['model1']), move next_token2 to that device
                    current_ids1 = torch.cat([current_ids1, next_token2.to(self.device_map['model1'])], dim=1)
                elif feed_mode == "swap":
                    current_text1 += token_text2
                    current_text2 += token_text1
                    # For current_ids1, use next_token2 from model2 but move it to device_map['model1']
                    current_ids1 = torch.cat([current_ids1, next_token2.to(self.device_map['model1'])], dim=1)
                    # For current_ids2, use next_token1 from model1 but move it to device_map['model2']
                    current_ids2 = torch.cat([current_ids2, next_token1.to(self.device_map['model2'])], dim=1)
                
                new_mask1 = torch.ones(current_ids1.size(0), 1, device=self.device_map['model1'])
                new_mask2 = torch.ones(current_ids2.size(0), 1, device=self.device_map['model2'])
                attention_mask1 = torch.cat([attention_mask1, new_mask1], dim=1)
                attention_mask2 = torch.cat([attention_mask2, new_mask2], dim=1)
                
                elapsed = time.time() - start_time
                tokens_generated = step + 1
                speed = tokens_generated / elapsed if elapsed > 0 else 0.0
                status = f"Step {step+1}/{max_length} - Generating... ({elapsed:.2f}s, {speed:.2f} tokens/sec)"
                
                yield current_text1, current_text2, status
                
                eos1 = (next_token1 == tokenizer1.eos_token_id).any()
                eos2 = (next_token2 == tokenizer2.eos_token_id).any()
                if eos1 and eos2:
                    break
        
        yield current_text1, current_text2, f"Generation completed in {time.time()-start_time:.2f} seconds"
    
    def stream_responses(self, input1, input2, max_length, temperature, do_sample, seed, feed_mode, progress=gr.Progress()):
        """Generate responses with streaming updates, including feed mode."""
        seed_val = int(seed) if seed and seed.strip() != "" else None
        input1 = input1.strip()
        input2 = input2.strip() if not self.joint_mode else input1.strip()
        torch.cuda.empty_cache()
        print(f"Input1: {input1[:50]}...")
        print(f"Input2: {input2[:50]}...")
        tokens1 = self.tokenizer1(input1, return_tensors="pt")
        tokens2 = self.tokenizer2(input2, return_tensors="pt")
        print(f"Input1 tokens: {tokens1['input_ids'].shape}")
        print(f"Input2 tokens: {tokens2['input_ids'].shape}")
        decoded1 = self.tokenizer1.decode(tokens1["input_ids"][0])
        decoded2 = self.tokenizer2.decode(tokens2["input_ids"][0])
        print(f"Decoded input1: {decoded1[:50]}...")
        print(f"Decoded input2: {decoded2[:50]}...")
        
        for out1, out2, status in self.generate_dual_streaming(
                tokens1["input_ids"],
                tokens2["input_ids"],
                self.tokenizer1,
                self.tokenizer2,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                seed=seed_val,
                feed_mode=feed_mode
            ):
            progress(0, desc=status)
            yield out1, out2, status

    def create_ui(self):
        """Create the Gradio UI for dual model inference with streaming and feed mode switching."""
        with gr.Blocks(title="Dual Model Generator") as app:
            gr.Markdown("# Dual Model Generator")
            gr.Markdown("Generate text using two models with cross-attention fusion, token generation speed display, and customizable feed mode.")
            
            with gr.Row():
                joint_mode_btn = gr.Button("Toggle Joint Mode", variant="primary")
                joint_mode_indicator = gr.Textbox(value="Joint Mode: ON", label="Mode", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Model 1 (Qwen)")
                    input1 = gr.Textbox(lines=5, label="Input Prompt (Model 1)", 
                                        placeholder="Enter prompt for Model 1...")
                    output1 = gr.Textbox(lines=10, label="Generated Output (Model 1)")
                with gr.Column():
                    gr.Markdown("### Model 2 (Llama)")
                    input2 = gr.Textbox(lines=5, label="Input Prompt (Model 2)",
                                        placeholder="Enter prompt for Model 2...")
                    output2 = gr.Textbox(lines=10, label="Generated Output (Model 2)")
            
            with gr.Row():
                max_length = gr.Slider(minimum=10, maximum=512, value=100, step=10, label="Max New Tokens")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
            
            with gr.Row():
                do_sample = gr.Checkbox(value=True, label="Use Sampling")
                seed = gr.Textbox(value="42", label="Seed (empty for random)")
                random_seed_btn = gr.Button("Random Seed", size="sm")
            
            with gr.Row():
                feed_mode = gr.Dropdown(
                    choices=["default", "model1_both", "model2_both", "swap"],
                    value="default",
                    label="Feed Mode",
                    info="Select which model's generated token feeds into which model."
                )
            
            with gr.Row():
                generate_btn = gr.Button("Generate", variant="primary", size="lg")
                clear_btn = gr.Button("Clear Outputs", size="lg")
            
            with gr.Row():
                status = gr.Textbox(label="Status", interactive=False)
            
            joint_mode_btn.click(fn=self.toggle_joint_mode, outputs=joint_mode_indicator)
            random_seed_btn.click(fn=lambda: str(random.randint(0, 999999)), outputs=seed)
            input1.change(fn=lambda x, y: self.sync_inputs(x, y, "input1"), inputs=[input1, input2], outputs=[input1, input2])
            input2.change(fn=lambda x, y: self.sync_inputs(x, y, "input2"), inputs=[input2, input1], outputs=[input2, input1])
            generate_btn.click(fn=self.stream_responses,
                               inputs=[input1, input2, max_length, temperature, do_sample, seed, feed_mode],
                               outputs=[output1, output2, status])
            clear_btn.click(fn=lambda: ("", "", "Outputs cleared"), outputs=[output1, output2, status])
            gr.Examples(
                examples=[
                    ["Write a short story about a robot learning emotions.", "Write a short story about a robot learning emotions."],
                    ["Explain quantum computing in simple terms.", "Write a poem about quantum computing."],
                    ["Write step-by-step instructions for making chocolate chip cookies.", "Describe the history of chocolate chip cookies."],
                ],
                inputs=[input1, input2],
            )
        return app

def main():
    config = {
        "model_dir1": "./DeepSeek-R1-Distill-Qwen-1.5B",
        "model_dir2": "./Llama-3.2-1B",
        "fusion_output_dim": 2048,
        "checkpoint_path": "./model_outputs/dual_model_best.pt"
    }
    
    inference_app = DualModelInferenceApp(config["checkpoint_path"], config)
    app = inference_app.create_ui()
    app.launch(share=False)

if __name__ == "__main__":
    main()

