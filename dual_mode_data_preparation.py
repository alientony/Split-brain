import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ----- Enhanced Fusion Layer with Cross-Attention -----

class EnhancedFusionLayer(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, fusion_output_dim):
        super().__init__()  # Fixed syntax error by removing trailing 'c'
        # Dimensionality
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.fusion_output_dim = fusion_output_dim
        
        # Cross-attention components
        # Query, Key, Value projections for cross-attention
        self.query1 = nn.Linear(hidden_dim1, fusion_output_dim)
        self.key2 = nn.Linear(hidden_dim2, fusion_output_dim)
        self.value2 = nn.Linear(hidden_dim2, fusion_output_dim)
        
        self.query2 = nn.Linear(hidden_dim2, fusion_output_dim)
        self.key1 = nn.Linear(hidden_dim1, fusion_output_dim)
        self.value1 = nn.Linear(hidden_dim1, fusion_output_dim)
        
        # Output projections
        self.output1 = nn.Linear(fusion_output_dim, fusion_output_dim)
        self.output2 = nn.Linear(fusion_output_dim, fusion_output_dim)
        
        # Gating mechanism
        self.gate1 = nn.Linear(hidden_dim1 + hidden_dim2, fusion_output_dim)
        self.gate2 = nn.Linear(hidden_dim1 + hidden_dim2, fusion_output_dim)
        
        # Fusion for combined representation
        self.fusion_linear = nn.Linear(hidden_dim1 + hidden_dim2, fusion_output_dim)
        
        # Activations
        self.activation = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_output_dim)
        self.norm2 = nn.LayerNorm(fusion_output_dim)
        
        # Scaling factor for attention
        self.scale = torch.sqrt(torch.tensor(fusion_output_dim, dtype=torch.float))

    def cross_attention(self, query, key, value, mask=None):
        # Calculate attention scores - ensure all tensors are on the same device
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is properly shaped for broadcasting
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
                
            # Use a smaller negative value that's safe for float16
            # float16 has a min value of approximately -65504
            mask_value = -1e4  # Much safer for float16 than -1e9
            attention_scores = attention_scores.masked_fill(mask == 0, mask_value)
        
        # Apply softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        context = torch.matmul(attention_weights, value)
        
        return context, attention_weights

    def forward(self, x1, x2, mask1=None, mask2=None, mode="auto"):
        # Ensure all inputs are on the same device
        device = x1.device
        batch_size, seq_len1, _ = x1.shape
        _, seq_len2, _ = x2.shape
        
        # Determine processing mode (same prompt or different prompts)
        if mode == "auto":
            # Auto-detection logic can be implemented here if needed
            # For now, we'll always use the full cross-attention approach
            mode = "cross"
        
        # 1. Process with cross-attention
        # Model1 attends to Model2
        q1 = self.query1(x1)  # [batch, seq1, fusion_dim]
        k2 = self.key2(x2)    # [batch, seq2, fusion_dim]
        v2 = self.value2(x2)  # [batch, seq2, fusion_dim]
        
        # Handle masking for attention
        attn_mask2 = None
        if mask2 is not None:
            # Create attention mask for model2
            attn_mask2 = mask2.to(device)
        
        # Apply cross-attention with proper masking
        context1_2, attn1_2 = self.cross_attention(q1, k2, v2, attn_mask2)
        
        # Model2 attends to Model1
        q2 = self.query2(x2)  # [batch, seq2, fusion_dim]
        k1 = self.key1(x1)    # [batch, seq1, fusion_dim]
        v1 = self.value1(x1)  # [batch, seq1, fusion_dim]
        
        # Handle masking for attention
        attn_mask1 = None
        if mask1 is not None:
            # Create attention mask for model1
            attn_mask1 = mask1.to(device)
        
        # Apply cross-attention with proper masking
        context2_1, attn2_1 = self.cross_attention(q2, k1, v1, attn_mask1)
        
        # 2. Process outputs
        out1 = self.output1(context1_2)  # Model1's representation after attending to Model2
        out2 = self.output2(context2_1)  # Model2's representation after attending to Model1
        
        # Create projection matrices for residual connections
        # Use more controlled approach to handle dimension mismatches
        x1_proj = None
        if self.hidden_dim1 != self.fusion_output_dim:
            x1_proj = self.proj1(x1) if hasattr(self, 'proj1') else nn.functional.linear(
                x1, self.output1.weight[:, :self.hidden_dim1])
        else:
            x1_proj = x1
            
        x2_proj = None
        if self.hidden_dim2 != self.fusion_output_dim:
            x2_proj = self.proj2(x2) if hasattr(self, 'proj2') else nn.functional.linear(
                x2, self.output2.weight[:, :self.hidden_dim2])
        else:
            x2_proj = x2
        
        # Apply layer norm with residual connections
        out1 = self.norm1(out1 + x1_proj)
        out2 = self.norm2(out2 + x2_proj)
        
        # 3. Gating mechanism
        # Concatenate for fusion and gating
        concat = torch.cat((x1, x2), dim=-1)
        
        # Compute the main fused representation
        fused = self.activation(self.fusion_linear(concat))
        
        # Compute gates (adaptive weighting for each model's contribution)
        gate1 = self.sigmoid(self.gate1(concat))
        gate2 = self.sigmoid(self.gate2(concat))
        
        # 4. Return all the representations
        return {
            "fused": fused,           # Combined representation with simple fusion
            "out1": out1,             # Model1's representation after attending to Model2
            "out2": out2,             # Model2's representation after attending to Model1
            "gate1": gate1,           # Gate values for Model1
            "gate2": gate2,           # Gate values for Model2
            "attn1_2": attn1_2,       # Attention weights: Model1 attending to Model2
            "attn2_1": attn2_1        # Attention weights: Model2 attending to Model1
        }

# ----- Dual Decoder Model with Multi-Task Support -----

class DualDecoderModel(nn.Module):
    def __init__(self, model1, model2, tokenizer1, tokenizer2, fusion_output_dim, 
                 freeze_base_models=True, device_map=None):
        """
        Enhanced DualDecoderModel with multi-task capabilities and balanced GPU memory usage.
        
        Args:
            model1: First base model (Qwen)
            model2: Second base model (Llama)
            tokenizer1: Tokenizer for first model
            tokenizer2: Tokenizer for second model
            fusion_output_dim: Dimension for the fusion representations
            freeze_base_models: Whether to freeze base model parameters
            device_map: Dictionary mapping model components to devices
        """
        super().__init__()
        
        # Device mapping for model parallelism
        self.device_map = device_map if device_map else {}
        self.device1 = self.device_map.get('model1', 'cuda:0')
        self.device2 = self.device_map.get('model2', 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
        self.fusion_device = self.device_map.get('fusion', 'cuda:1')
        
        # NEW: Additional device for memory balancing
        self.extra_device = self.device_map.get('extra', self.device1)  # Default to model1's device
        
        # Store models without immediately moving them to devices
        # We'll explicitly move data during forward pass
        self.model1 = model1
        self.model2 = model2
        
        # Store vocab sizes
        self.vocab_size1 = len(tokenizer1)
        self.vocab_size2 = len(tokenizer2)
        
        # Store embedding dimensions
        self.hidden_size1 = model1.config.hidden_size
        self.hidden_size2 = model2.config.hidden_size
        
        # Enhanced fusion layer with cross-attention
        self.fusion_layer = EnhancedFusionLayer(
            self.hidden_size1, 
            self.hidden_size2, 
            fusion_output_dim
        )
        
        # Intermediate projection layers for dimension matching if needed
        self.proj1 = nn.Linear(self.hidden_size1, fusion_output_dim)
        self.proj2 = nn.Linear(self.hidden_size2, fusion_output_dim)
        
        # Language model heads - DISTRIBUTED ACROSS BOTH GPUs
        # 1. Main fused output for both vocabularies - one on each GPU
        self.fused_lm_head1 = nn.Linear(fusion_output_dim, self.vocab_size1, bias=False)  # For model1 vocabulary
        self.fused_lm_head2 = nn.Linear(fusion_output_dim, self.vocab_size2, bias=False)  # For model2 vocabulary
        
        # 2. Model-specific heads - one on each GPU  
        self.lm_head1 = nn.Linear(fusion_output_dim, self.vocab_size1, bias=False)
        self.lm_head2 = nn.Linear(fusion_output_dim, self.vocab_size2, bias=False)
        
        # Task type classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_size1 + self.hidden_size2, fusion_output_dim),
            nn.GELU(),
            nn.Linear(fusion_output_dim, 1),
            nn.Sigmoid()
        )
        
        # Move models to their respective devices
        self.model1.to(self.device1)
        self.model2.to(self.device2)
        
        # NEW: Distribute computation across both GPUs
        self.fusion_layer.to(self.fusion_device)
        self.proj1.to(self.fusion_device)
        self.proj2.to(self.fusion_device)
        self.task_classifier.to(self.fusion_device)
        
        # BALANCED PLACEMENT: Split the heads between GPUs
        self.fused_lm_head1.to(self.fusion_device)  # Keep on fusion device
        self.fused_lm_head2.to(self.extra_device)   # Move to the other GPU
        self.lm_head1.to(self.fusion_device)        # Keep on fusion device 
        self.lm_head2.to(self.extra_device)         # Move to the other GPU
        
        if freeze_base_models:
            for param in self.model1.parameters():
                param.requires_grad = False
            for param in self.model2.parameters():
                param.requires_grad = False

        # Print confirmation of initialization with memory balancing
        print(f"Initialized dual model with balanced memory distribution across GPUs:")
        print(f"  Model1: {self.device1}, Model2: {self.device2}")
        print(f"  Fusion layer: {self.fusion_device}")
        print(f"  Heads for Model1 (vocab size {self.vocab_size1}): {self.fusion_device}")
        print(f"  Heads for Model2 (vocab size {self.vocab_size2}): {self.extra_device}")

    def forward(self, input_ids1, input_ids2, attention_mask1=None, attention_mask2=None, 
                labels1=None, labels2=None, mode="auto", return_details=False):
        """Forward pass with memory-optimized tensor movement and balanced loss calculation"""
        # Move inputs to their respective devices
        input_ids1 = input_ids1.to(self.device1)
        attention_mask1 = attention_mask1.to(self.device1) if attention_mask1 is not None else None
        
        input_ids2 = input_ids2.to(self.device2)
        attention_mask2 = attention_mask2.to(self.device2) if attention_mask2 is not None else None
        
        # Process through base models with no_grad if models are frozen
        with torch.set_grad_enabled(not self.model1.training or next(self.model1.parameters()).requires_grad):
            outputs1 = self.model1(input_ids=input_ids1, attention_mask=attention_mask1)
            
        with torch.set_grad_enabled(not self.model2.training or next(self.model2.parameters()).requires_grad):
            outputs2 = self.model2(input_ids=input_ids2, attention_mask=attention_mask2)
        
        # Extract hidden states
        if hasattr(outputs1, "last_hidden_state"):
            hidden1 = outputs1.last_hidden_state
        else:
            hidden1 = outputs1[0]
            
        if hasattr(outputs2, "last_hidden_state"):
            hidden2 = outputs2.last_hidden_state
        else:
            hidden2 = outputs2[0]
        
        # Move hidden states to fusion device for processing
        hidden1 = hidden1.to(self.fusion_device)
        hidden2 = hidden2.to(self.fusion_device)
        
        # Determine task type (are inputs the same or different?)
        if mode == "auto":
            cls1 = hidden1[:, 0, :]  # First token representation from model1
            cls2 = hidden2[:, 0, :]  # First token representation from model2
            task_input = torch.cat([cls1, cls2], dim=-1)
            task_prob = self.task_classifier(task_input)
            detected_mode = "multi" if torch.mean(task_prob) > 0.5 else "single"
        else:
            detected_mode = mode
        
        # Move attention masks to fusion device if they exist
        attn_mask1_fusion = attention_mask1.to(self.fusion_device) if attention_mask1 is not None else None
        attn_mask2_fusion = attention_mask2.to(self.fusion_device) if attention_mask2 is not None else None
        
        # Process with enhanced fusion layer
        fusion_outputs = self.fusion_layer(
            hidden1, 
            hidden2, 
            mask1=attn_mask1_fusion,
            mask2=attn_mask2_fusion,
            mode=detected_mode
        )
        
        # Extract representations
        fused = fusion_outputs["fused"]  # Main fused representation
        out1 = fusion_outputs["out1"]    # Model1 representation after cross-attention
        out2 = fusion_outputs["out2"]    # Model2 representation after cross-attention
        gate1 = fusion_outputs["gate1"]  # Gating values for model1
        gate2 = fusion_outputs["gate2"]  # Gating values for model2
        
        # MEMORY OPTIMIZATION: Process Model1 outputs on fusion_device, Model2 on extra_device
        # Generate logits for Model1 on fusion device
        fused_logits1 = self.fused_lm_head1(fused)
        logits1 = self.lm_head1(out1)
        
        # Move tensors to the extra device for Model2 processing
        fused_extra = fused.to(self.extra_device)
        out2_extra = out2.to(self.extra_device)
        
        # Generate logits for Model2 on extra device
        fused_logits2 = self.fused_lm_head2(fused_extra)
        logits2 = self.lm_head2(out2_extra)
        
        # Calculate losses if labels are provided
        loss = None
        loss1 = None
        loss2 = None
        total_loss = None
        
        if labels1 is not None:
            labels1 = labels1.to(self.fusion_device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Process Model1 losses on fusion device
            if fused_logits1.size(1) > 1 and labels1.size(1) > 1:
                shift_fused_logits1 = fused_logits1[..., :-1, :].contiguous()
                shift_logits1 = logits1[..., :-1, :].contiguous()
                shift_labels1 = labels1[..., 1:].contiguous()
                
                if detected_mode == "single":
                    loss = loss_fct(shift_fused_logits1.view(-1, shift_fused_logits1.size(-1)), 
                                    shift_labels1.view(-1))
                
                loss1 = loss_fct(shift_logits1.view(-1, shift_logits1.size(-1)), 
                                shift_labels1.view(-1))
            else:
                if detected_mode == "single":
                    loss = torch.tensor(0.0, requires_grad=True, device=self.fusion_device)
                loss1 = torch.tensor(0.0, requires_grad=True, device=self.fusion_device) 
        
        if labels2 is not None:
            # Move labels2 to extra device for processing
            labels2 = labels2.to(self.extra_device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Process Model2 losses on extra device
            if fused_logits2.size(1) > 1 and labels2.size(1) > 1:
                shift_fused_logits2 = fused_logits2[..., :-1, :].contiguous()
                shift_logits2 = logits2[..., :-1, :].contiguous()
                shift_labels2 = labels2[..., 1:].contiguous()
                
                # Calculate loss for model2 on extra device
                loss2_raw = loss_fct(shift_logits2.view(-1, shift_logits2.size(-1)), 
                                  shift_labels2.view(-1))
                
                # Move loss back to fusion device if needed
                loss2 = loss2_raw.to(self.fusion_device)
                
                # Calculate fused loss for model2 in single mode
                if detected_mode == "single" and loss is None:
                    fused_loss2 = loss_fct(shift_fused_logits2.view(-1, shift_fused_logits2.size(-1)),
                                         shift_labels2.view(-1))
                    loss = fused_loss2.to(self.fusion_device)
            else:
                loss2 = torch.tensor(0.0, requires_grad=True, device=self.fusion_device)
        
        # MODIFIED: Determine final combined loss with balanced weighting to penalize imbalance
        if loss1 is not None and loss2 is not None:
            # Calculate relative difference between the losses to detect imbalance
            # We add a small epsilon to prevent division by zero
            epsilon = 1e-6
            
            # Ensure loss values are positive for comparison
            abs_loss1 = torch.abs(loss1) + epsilon
            abs_loss2 = torch.abs(loss2) + epsilon
            
            # Calculate ratio of larger loss to smaller loss (always >= 1.0)
            ratio = torch.max(abs_loss1 / abs_loss2, abs_loss2 / abs_loss1)
            
            # Define a scaling factor that increases quadratically with the ratio
            # This heavily penalizes large imbalances between the two models
            imbalance_penalty = torch.clamp(ratio * ratio, min=1.0, max=10.0)
            
            # Figure out which loss is higher to apply the penalty to
            if abs_loss1 > abs_loss2:
                # Model1 has higher loss - increase its weight in the total loss
                weight1 = 0.5 * imbalance_penalty
                weight2 = 0.5
            else:
                # Model2 has higher loss - increase its weight in the total loss
                weight1 = 0.5
                weight2 = 0.5 * imbalance_penalty
            
            # Normalize weights to sum to 1.0 for stable training
            weight_sum = weight1 + weight2
            weight1 = weight1 / weight_sum
            weight2 = weight2 / weight_sum
            
            # Calculate final combined loss with rebalanced weights
            if detected_mode == "single":
                if loss is not None:
                    # Include the general fusion loss with a constant weight
                    total_loss = 0.3 * loss + weight1 * loss1 + weight2 * loss2
                else:
                    total_loss = weight1 * loss1 + weight2 * loss2
            else:
                if loss is not None:
                    total_loss = 0.2 * loss + weight1 * loss1 + weight2 * loss2
                else:
                    total_loss = weight1 * loss1 + weight2 * loss2
                    
            # Log the imbalance and weights for monitoring (during training)
            if self.training and torch.rand(1).item() < 0.01:
                print(f"Loss balance - L1: {loss1.item():.4f}, L2: {loss2.item():.4f}, "
                      f"Ratio: {ratio.item():.2f}, Penalty: {imbalance_penalty.item():.2f}, "
                      f"Weights: {weight1.item():.2f}/{weight2.item():.2f}")
        elif loss1 is not None:
            total_loss = loss1
        elif loss2 is not None:
            total_loss = loss2
        elif loss is not None:
            total_loss = loss
        
        # For returning, move logits2 back to fusion device if needed
        if return_details and self.fusion_device != self.extra_device:
            fused_logits2 = fused_logits2.to(self.fusion_device)
            logits2 = logits2.to(self.fusion_device)
            gate2 = gate2.to(self.fusion_device)
        
        if return_details:
            return {
                "loss": total_loss,
                "fused_logits1": fused_logits1,
                "fused_logits2": fused_logits2,
                "logits1": logits1,
                "logits2": logits2,
                "gate1": gate1,
                "gate2": gate2,
                "mode": detected_mode,
                "attn1_2": fusion_outputs["attn1_2"],
                "attn2_1": fusion_outputs["attn2_1"]
            }
        else:
            return {
                "loss": total_loss,
                "fused_logits1": fused_logits1,
                "fused_logits2": fused_logits2,
                "logits1": logits1,
                "logits2": logits2,
                "mode": detected_mode
            }

    def generate_dual(self, input_ids1, input_ids2, tokenizer1, tokenizer2, 
                     max_length=100, temperature=1.0, do_sample=True, 
                     attention_mask1=None, attention_mask2=None,
                     return_attention_maps=False):
        """Memory-optimized generation function that distributes computation across GPUs"""
        self.eval()
        batch_size = input_ids1.size(0)
        
        # Move inputs to the correct devices
        input_ids1 = input_ids1.to(self.device1)
        input_ids2 = input_ids2.to(self.device2)
        
        # Initialize tracking variables
        current_ids1 = input_ids1.clone()
        current_ids2 = input_ids2.clone()
        
        # Create attention masks if not provided
        if attention_mask1 is None:
            attention_mask1 = torch.ones_like(current_ids1, device=self.device1)
        else:
            attention_mask1 = attention_mask1.to(self.device1)
        
        if attention_mask2 is None:
            attention_mask2 = torch.ones_like(current_ids2, device=self.device2)
        else:
            attention_mask2 = attention_mask2.to(self.device2)
            
        # Track which sequences have finished generating
        finished1 = torch.zeros(batch_size, dtype=torch.bool, device=self.fusion_device)
        finished2 = torch.zeros(batch_size, dtype=torch.bool, device=self.extra_device)
        
        # Determine if inputs are identical (for mode detection)
        inputs_match = torch.equal(input_ids1.cpu(), input_ids2.cpu()) if input_ids1.size() == input_ids2.size() else False
        mode = "single" if inputs_match else "multi"
        
        # For storing attention maps if requested
        attention_maps = [] if return_attention_maps else None
        
        # Add gate projection layers if they don't exist yet
        if not hasattr(self, 'gate1_proj'):
            print("Creating gate projection layer for model 1")
            self.gate1_proj = nn.Linear(self.fusion_layer.fusion_output_dim, self.vocab_size1).to(self.fusion_device)
        
        if not hasattr(self, 'gate2_proj'):
            print("Creating gate projection layer for model 2")
            self.gate2_proj = nn.Linear(self.fusion_layer.fusion_output_dim, self.vocab_size2).to(self.extra_device)
        
        with torch.no_grad():
            # Generation loop
            for step in range(max_length):
                # Run forward pass
                outputs = self.forward(
                    input_ids1=current_ids1, 
                    input_ids2=current_ids2, 
                    attention_mask1=attention_mask1, 
                    attention_mask2=attention_mask2,
                    mode=mode,
                    return_details=True  # Always get full details for generation
                )
                
                # Store attention maps if requested
                if return_attention_maps:
                    attention_maps.append({
                        "attn1_2": outputs["attn1_2"].detach().cpu(),
                        "attn2_1": outputs["attn2_1"].detach().cpu(),
                    })
                
                # Extract logits and gates
                fused_logits1 = outputs["fused_logits1"]
                fused_logits2 = outputs["fused_logits2"]
                logits1 = outputs["logits1"]
                logits2 = outputs["logits2"]
                gate1 = outputs["gate1"]
                gate2 = outputs["gate2"]
                detected_mode = outputs["mode"]
                
                # Get only the last token logits for generation
                # Use proper dimension handling to ensure we get the right shape
                fused_logits1_last = fused_logits1[:, -1, :] if len(fused_logits1.shape) > 2 else fused_logits1
                fused_logits2_last = fused_logits2[:, -1, :] if len(fused_logits2.shape) > 2 else fused_logits2
                logits1_last = logits1[:, -1, :] if len(logits1.shape) > 2 else logits1
                logits2_last = logits2[:, -1, :] if len(logits2.shape) > 2 else logits2
                
                # MEMORY OPTIMIZATION: Process on separate devices
                # Keep model1 processing on fusion device
                fused_logits1_last = fused_logits1_last.to(self.fusion_device)
                logits1_last = logits1_last.to(self.fusion_device)
                
                # Move model2 processing to extra device
                fused_logits2_last = fused_logits2_last.to(self.extra_device)
                logits2_last = logits2_last.to(self.extra_device)
                
                # Apply balanced weighting for both models
                if detected_mode == "single":
                    # In single-task mode, the fused representation is more important
                    weighted_logits1 = (0.7 * fused_logits1_last + 0.3 * logits1_last) / 1.0
                    weighted_logits2 = (0.7 * fused_logits2_last + 0.3 * logits2_last) / 1.0
                else:
                    # In multi-task mode, the separate decoders are more important
                    weighted_logits1 = (0.4 * fused_logits1_last + 0.6 * logits1_last) / 1.0
                    weighted_logits2 = (0.4 * fused_logits2_last + 0.6 * logits2_last) / 1.0
                
                # Apply gating - move gates to appropriate devices for computation
                gate1_last = gate1[:, -1, :].to(self.fusion_device)
                gate2_last = gate2[:, -1, :].to(self.extra_device)
                
                # Project gates to vocabulary space - on separate devices
                gate1_proj = self.gate1_proj(gate1_last)  # On fusion device
                gate2_proj = self.gate2_proj(gate2_last)  # On extra device
                
                # Apply sigmoid to ensure values are between 0 and 1
                gate1_proj = torch.sigmoid(gate1_proj)
                gate2_proj = torch.sigmoid(gate2_proj)
                
                # Apply projected gates to logits
                weighted_logits1 = weighted_logits1 * gate1_proj  # On fusion device
                weighted_logits2 = weighted_logits2 * gate2_proj  # On extra device
                
                # Apply temperature
                if temperature > 0:
                    weighted_logits1 = weighted_logits1 / temperature
                    weighted_logits2 = weighted_logits2 / temperature
                
                # Apply sampling or greedy decoding - on separate devices
                if do_sample:
                    # Handle any NaN/Inf values
                    weighted_logits1 = torch.nan_to_num(weighted_logits1)
                    weighted_logits2 = torch.nan_to_num(weighted_logits2)
                    
                    # Apply softmax for sampling
                    probs1 = torch.softmax(weighted_logits1, dim=-1)
                    probs2 = torch.softmax(weighted_logits2, dim=-1)
                    
                    # Debug info - print shape before sampling
                    if step == 0:
                        print(f"Probs1 shape before sampling: {probs1.shape}")
                        print(f"Probs2 shape before sampling: {probs2.shape}")
                    
                    # Fix dimensions for multinomial - multinomial expects [batch, vocab] exactly
                    # If we have extra dimensions, reshape correctly
                    if len(probs1.shape) > 2:
                        # Reshape to [batch*seq, vocab]
                        probs1 = probs1.reshape(-1, probs1.size(-1))
                        print(f"Reshaped probs1 to: {probs1.shape}")
                    elif len(probs1.shape) == 1:
                        # Add batch dimension if just 1D
                        probs1 = probs1.unsqueeze(0)
                        print(f"Added batch dim to probs1: {probs1.shape}")
                    
                    if len(probs2.shape) > 2:
                        probs2 = probs2.reshape(-1, probs2.size(-1))
                        print(f"Reshaped probs2 to: {probs2.shape}")
                    elif len(probs2.shape) == 1:
                        probs2 = probs2.unsqueeze(0)
                        print(f"Added batch dim to probs2: {probs2.shape}")
                    
                    # Check for NaNs that could break multinomial
                    if torch.isnan(probs1).any():
                        print("WARNING: NaN detected in probs1, fixing...")
                        probs1 = torch.nan_to_num(probs1)
                        # Ensure it still sums to 1
                        probs1 = probs1 / probs1.sum(dim=-1, keepdim=True)
                    
                    if torch.isnan(probs2).any():
                        print("WARNING: NaN detected in probs2, fixing...")
                        probs2 = torch.nan_to_num(probs2)
                        # Ensure it still sums to 1
                        probs2 = probs2 / probs2.sum(dim=-1, keepdim=True)
                    
                    # Additional safety checks for multinomial sampling
                    if not torch.all(torch.isfinite(probs1)):
                        print("WARNING: Non-finite values in probs1, using greedy decoding")
                        next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                    else:
                        try:
                            # Sample
                            next_token1 = torch.multinomial(probs1, num_samples=1)
                        except RuntimeError as e:
                            print(f"Error during sampling from probs1: {e}")
                            print(f"probs1 shape: {probs1.shape}, sum: {probs1.sum().item()}")
                            print("Falling back to greedy decoding")
                            next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                    
                    if not torch.all(torch.isfinite(probs2)):
                        print("WARNING: Non-finite values in probs2, using greedy decoding")
                        next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
                    else:
                        try:
                            next_token2 = torch.multinomial(probs2, num_samples=1)
                        except RuntimeError as e:
                            print(f"Error during sampling from probs2: {e}")
                            print(f"probs2 shape: {probs2.shape}, sum: {probs2.sum().item()}")
                            print("Falling back to greedy decoding")
                            next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
                else:
                    # Greedy decoding on separate devices
                    next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                    next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
                
                # Ensure next_tokens have the right shape
                if len(next_token1.shape) == 0:
                    next_token1 = next_token1.unsqueeze(0)
                if len(next_token2.shape) == 0:
                    next_token2 = next_token2.unsqueeze(0)
                
                # Update finished status based on EOS tokens - on appropriate devices
                if step == 0:
                    print(f"Next token shapes - token1: {next_token1.shape}, token2: {next_token2.shape}")
                
                # Ensure finished tensors have compatible shape for comparison
                if next_token1.shape != finished1.shape:
                    if len(next_token1.shape) > len(finished1.shape):
                        # Either squeeze or pick first element
                        if next_token1.size(0) == 1:
                            next_token1 = next_token1.squeeze(0)
                        else:
                            next_token1 = next_token1[0]
                    else:
                        # Add dimensions to match
                        while len(next_token1.shape) < len(finished1.shape):
                            next_token1 = next_token1.unsqueeze(0)
                
                if next_token2.shape != finished2.shape:
                    if len(next_token2.shape) > len(finished2.shape):
                        if next_token2.size(0) == 1:
                            next_token2 = next_token2.squeeze(0)
                        else:
                            next_token2 = next_token2[0]
                    else:
                        while len(next_token2.shape) < len(finished2.shape):
                            next_token2 = next_token2.unsqueeze(0)
                
                # Check for EOS tokens
                eos1 = torch.eq(next_token1, tokenizer1.eos_token_id)
                eos2 = torch.eq(next_token2, tokenizer2.eos_token_id)
                
                finished1 = finished1 | eos1.to(self.fusion_device)
                finished2 = finished2 | eos2.to(self.extra_device)
                
                # Only update tokens for sequences that aren't finished
                next_token1 = next_token1.masked_fill(finished1, tokenizer1.pad_token_id)
                next_token2 = next_token2.masked_fill(finished2, tokenizer2.pad_token_id)
                
                # Move tokens to appropriate input devices for next iteration
                next_token1 = next_token1.to(self.device1)
                next_token2 = next_token2.to(self.device2)
                
                # Ensure tokens have the right shape for concatenation
                if len(next_token1.shape) == 1:
                    next_token1 = next_token1.unsqueeze(-1)
                if len(next_token2.shape) == 1:
                    next_token2 = next_token2.unsqueeze(-1)
                
                # Add next tokens to sequences
                current_ids1 = torch.cat([current_ids1, next_token1], dim=-1)
                current_ids2 = torch.cat([current_ids2, next_token2], dim=-1)
                
                # Update attention masks
                attention_mask1 = torch.cat([
                    attention_mask1, 
                    (~finished1).to(self.device1).long().unsqueeze(-1)
                ], dim=-1)
                
                attention_mask2 = torch.cat([
                    attention_mask2, 
                    (~finished2).to(self.device2).long().unsqueeze(-1)
                ], dim=-1)
                
                # Break if all sequences are finished
                if torch.all(finished1) and torch.all(finished2):
                    break
                    
        # Move tensors to CPU for decoding
        current_ids1_cpu = current_ids1.detach().cpu()
        current_ids2_cpu = current_ids2.detach().cpu()
                
        # Decode sequences
        decoded_texts1 = tokenizer1.batch_decode(current_ids1_cpu, skip_special_tokens=True)
        decoded_texts2 = tokenizer2.batch_decode(current_ids2_cpu, skip_special_tokens=True)
        
        if return_attention_maps:
            return decoded_texts1, decoded_texts2, attention_maps
        else:
            return decoded_texts1, decoded_texts2

def generate_dual_streaming(self, input_ids1, input_ids2, tokenizer1, tokenizer2, 
                        max_length=100, temperature=0.7, do_sample=True, 
                        attention_mask1=None, attention_mask2=None,
                        return_attention_maps=False, seed=None):
    """Generation function that yields intermediate outputs for streaming with seed control."""
    self.dual_model.eval()
    
    # Set initial seed if provided
    if seed is not None:
        # Set all random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Using seed: {seed} for generation")
    else:
        # If no seed provided, use a truly random seed
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
    
    start_time = time.time()
    for step in range(max_length):
        # Set per-step seed if seed is provided for reproducible but diverse steps
        if seed is not None:
            # Save current random state
            rng_state = torch.get_rng_state()
            np_state = np.random.get_state()
            py_state = random.getstate()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state_all()
            
            # Set seed for this step (different for each step)
            step_seed = seed + step
            random.seed(step_seed)
            np.random.seed(step_seed)
            torch.manual_seed(step_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(step_seed)
        
        with torch.no_grad():
            try:
                # Validate input shapes before processing
                if step > 0 and (attention_mask1.size(1) != current_ids1.size(1) or 
                                attention_mask2.size(1) != current_ids2.size(1)):
                    print(f"Warning: Attention mask mismatch at step {step}")
                    # Recreate attention masks to match input_ids
                    attention_mask1 = torch.ones_like(current_ids1, device=self.device_map['model1'])
                    attention_mask2 = torch.ones_like(current_ids2, device=self.device_map['model2'])
                
                # Ensure all tokens are valid indices (within vocab size)
                max_id1 = self.tokenizer1.vocab_size - 1
                max_id2 = self.tokenizer2.vocab_size - 1
                
                if torch.any(current_ids1 > max_id1):
                    print(f"Warning: Invalid token IDs for model1, fixing...")
                    current_ids1 = torch.clamp(current_ids1, max=max_id1)
                
                if torch.any(current_ids2 > max_id2):
                    print(f"Warning: Invalid token IDs for model2, fixing...")
                    current_ids2 = torch.clamp(current_ids2, max=max_id2)
                
                # Process through base models
                outputs1 = self.dual_model.model1(input_ids=current_ids1, attention_mask=attention_mask1)
                outputs2 = self.dual_model.model2(input_ids=current_ids2, attention_mask=attention_mask2)
                
                hidden1 = outputs1.last_hidden_state if hasattr(outputs1, "last_hidden_state") else outputs1[0]
                hidden2 = outputs2.last_hidden_state if hasattr(outputs2, "last_hidden_state") else outputs2[0]
            except RuntimeError as e:
                # Error recovery code here (unchanged)
                print(f"Error during model forward pass: {e}")
                print("Attempting recovery...")
                # Emergency recovery - try with smaller context
                # (rest of recovery code unchanged)
                # ...
            
            hidden1 = hidden1.to(self.device_map['fusion'])
            hidden2 = hidden2.to(self.device_map['fusion'])
            
            # Handle sequence length padding if needed (unchanged)
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
            
            # Process through fusion layer and calculate model outputs
            fusion_outputs = self.dual_model.fusion_layer(
                hidden1, 
                hidden2, 
                mask1=attention_mask1_fusion,
                mask2=attention_mask2_fusion,
                mode=mode
            )
            
            # Get all representations from fusion layer
            fused = fusion_outputs["fused"]  # Main fused representation
            out1 = fusion_outputs["out1"]    # Model1 representation after cross-attention
            out2 = fusion_outputs["out2"]    # Model2 representation after cross-attention
            gate1 = fusion_outputs["gate1"]  # Gating values for model1
            gate2 = fusion_outputs["gate2"]  # Gating values for model2
            
            # Generate logits using different heads - UPDATED for balanced approach
            # 1. Fused representations projected to both vocabularies
            fused_logits1 = self.dual_model.fused_lm_head1(fused)  # Uses model1's vocabulary
            fused_logits2 = self.dual_model.fused_lm_head2(fused)  # Uses model2's vocabulary
            
            # 2. Model-specific outputs
            logits1 = self.dual_model.lm_head1(out1)  # Uses model1's vocabulary
            logits2 = self.dual_model.lm_head2(out2)  # Uses model2's vocabulary
            
            # Get only the last token logits for generation - FIXED to ensure consistent dimensions
            # Always extract just the last token from each tensor
            fused_logits1_last = fused_logits1[:, -1, :] if len(fused_logits1.shape) > 2 else fused_logits1
            fused_logits2_last = fused_logits2[:, -1, :] if len(fused_logits2.shape) > 2 else fused_logits2
            logits1_last = logits1[:, -1, :] if len(logits1.shape) > 2 else logits1
            logits2_last = logits2[:, -1, :] if len(logits2.shape) > 2 else logits2
            
            # Move all logits to fusion device
            fused_logits1_last = fused_logits1_last.to(self.device_map['fusion'])
            fused_logits2_last = fused_logits2_last.to(self.device_map['fusion'])
            logits1_last = logits1_last.to(self.device_map['fusion'])
            logits2_last = logits2_last.to(self.device_map['fusion'])
            
            # Print diagnostic information about shape in first step
            if step == 0:
                print(f"Last token logits shapes (after dimension fixing):")
                print(f"  fused_logits1_last: {fused_logits1_last.shape}")
                print(f"  fused_logits2_last: {fused_logits2_last.shape}")
                print(f"  logits1_last: {logits1_last.shape}")
                print(f"  logits2_last: {logits2_last.shape}")
            
            # Apply weighting based on mode - BALANCED for both models
            if mode == "single":
                # In single-task mode, fused representation is more important
                weighted_logits1 = (0.7 * fused_logits1_last + 0.3 * logits1_last) / 1.0
                weighted_logits2 = (0.7 * fused_logits2_last + 0.3 * logits2_last) / 1.0
            else:
                # In multi-task mode, separate decoders are more important
                weighted_logits1 = (0.4 * fused_logits1_last + 0.6 * logits1_last) / 1.0
                weighted_logits2 = (0.4 * fused_logits2_last + 0.6 * logits2_last) / 1.0
            
            # Apply temperature scaling
            if temperature > 0:
                weighted_logits1 = weighted_logits1 / temperature
                weighted_logits2 = weighted_logits2 / temperature
            
            # Add a diagnostic print to help with debugging
            if step == 0:  # Only print for the first step to avoid excessive output
                print(f"Mode: {mode}")
                print(f"Vocab sizes - model1: {weighted_logits1.size(-1)}, model2: {weighted_logits2.size(-1)}")
                print(f"Using balanced weighting for both models")
            
            # Handle sampling vs greedy decoding
            if do_sample:
                # Handle any NaN/Inf values
                weighted_logits1 = torch.nan_to_num(weighted_logits1)
                weighted_logits2 = torch.nan_to_num(weighted_logits2)
                
                # Apply softmax for sampling - ensure we apply it on the vocab dimension (dim=-1)
                probs1 = torch.softmax(weighted_logits1, dim=-1)
                probs2 = torch.softmax(weighted_logits2, dim=-1)
                
                # Sample
                try:
                    next_token1 = torch.multinomial(probs1, num_samples=1)
                    next_token2 = torch.multinomial(probs2, num_samples=1)
                except RuntimeError as e:
                    print(f"Error during sampling: {e}")
                    # Fallback to greedy sampling if multinomial fails
                    print("Falling back to greedy decoding")
                    next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                    next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
            else:
                # Greedy decoding
                next_token1 = torch.argmax(weighted_logits1, dim=-1, keepdim=True)
                next_token2 = torch.argmax(weighted_logits2, dim=-1, keepdim=True)
            
            # Restore random state if seed was provided
            if seed is not None:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(rng_state)
                if torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(cuda_rng_state)
            
            # Rest of the generation code (unchanged)
            if len(next_token1.shape) > 2:
                next_token1 = next_token1.view(next_token1.size(0), -1)
            if len(next_token2.shape) > 2:
                next_token2 = next_token2.view(next_token2.size(0), -1)
            
            next_token1 = next_token1.to(self.device_map['model1'])
            next_token2 = next_token2.to(self.device_map['model2'])
            
            current_ids1 = torch.cat([current_ids1, next_token1], dim=1)
            current_ids2 = torch.cat([current_ids2, next_token2], dim=1)
            attention_mask1 = torch.cat([attention_mask1, torch.ones(attention_mask1.size(0), 1, device=self.device_map['model1'])], dim=1)
            attention_mask2 = torch.cat([attention_mask2, torch.ones(attention_mask2.size(0), 1, device=self.device_map['model2'])], dim=1)
            
            # Process token decoding for each model
            token_text1 = tokenizer1.decode(next_token1[0], skip_special_tokens=True)
            token_text2 = tokenizer2.decode(next_token2[0], skip_special_tokens=True)
            
            # Diagnostic for the first few tokens to help debug
            if step < 5:  # Only print for first few steps
                print(f"Step {step} - Token1 ID: {next_token1[0].item()}, Text: '{token_text1}'")
                print(f"Step {step} - Token2 ID: {next_token2[0].item()}, Text: '{token_text2}'")
            
            # Special handling for Model 2 (Llama) - this helps fix the output
            if not token_text2 or token_text2.isspace() or token_text2.startswith("<"):
                # Try decoding without skipping special tokens
                token_text2 = tokenizer2.decode(next_token2[0], skip_special_tokens=False)
                # If still problematic, use a failsafe approach
                if not token_text2 or token_text2.isspace():
                    # Get the actual token from vocabulary as fallback
                    token_id = next_token2[0].item()
                    if token_id < len(tokenizer2.vocab):
                        token_text2 = tokenizer2.convert_ids_to_tokens(token_id)
                        print(f"Fallback token2 text: '{token_text2}'")
            
            current_text1 += token_text1
            current_text2 += token_text2
            
            elapsed = time.time() - start_time
            status = f"Step {step+1}/{max_length} - Generating... ({elapsed:.2f}s)"
            
            # Yield the current generated texts and status message
            yield current_text1, current_text2, status
            
            eos1 = (next_token1 == tokenizer1.eos_token_id).any()
            eos2 = (next_token2 == tokenizer2.eos_token_id).any()
            if eos1 and eos2:
                break
    
    yield current_text1, current_text2, f"Generation completed in {time.time()-start_time:.2f} seconds"

# ----- Train function also needs update for checkpoint saving with dual heads -----

def train_dual_model(model, train_dataloader, val_dataloader, optimizer, tokenizer1, tokenizer2,
                     epochs=3, patience=2, accumulation_steps=8, fp16=True,
                     output_dir="./outputs", device_map=None, print_interval=7000):
    """
    Train the dual decoder model with explicit tracking of GradScaler state to prevent
    multiple unscale_() calls.
    """
    import os
    import torch  
    import time
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device information for reporting
    device1 = device_map.get('model1', 'cuda:1') if device_map else 'cuda:1'
    device2 = device_map.get('model2', 'cuda:0' if torch.cuda.device_count() > 1 else 'cuda:0') if device_map else 'cuda:0'
    fusion_device = device_map.get('fusion', 'cuda:1') if device_map else 'cuda:0'
    
    print(f"Training with devices - Model1: {device1}, Model2: {device2}, Fusion: {fusion_device}")
    print(f"Will print sample outputs every {print_interval} batches for monitoring")
    
    # Initialize mixed precision training if requested and available
    scaler = None
    if fp16 and torch.cuda.is_available():
        from torch.amp import autocast, GradScaler
        scaler = GradScaler()
        print("Using mixed precision training (FP16)")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Verify model is properly distributed across devices
    print("\nVerifying model device placement:")
    print(f"Model1 device: {next(model.model1.parameters()).device}")
    print(f"Model2 device: {next(model.model2.parameters()).device}")
    print(f"Fusion layer device: {next(model.fusion_layer.parameters()).device}")
    print(f"Projection layers devices: {next(model.proj1.parameters()).device}, {next(model.proj2.parameters()).device}")
    print(f"Language model heads devices: {next(model.lm_head1.parameters()).device}, {next(model.lm_head2.parameters()).device}")
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0
        total_single_samples = 0
        total_multi_samples = 0
        optimizer.zero_grad()
        
        # Track metrics
        start_time = time.time()
        global_batch = 0  # Track overall batch count
        
        # IMPORTANT: Track GradScaler state to prevent multiple unscale_() calls
        need_optimizer_step = False
        gradients_accumulated = 0
        
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
            global_batch = epoch * len(train_dataloader) + i + 1
            
            # Get inputs
            input_ids1 = batch["input_ids1"]
            attention_mask1 = batch["attention_mask1"] if "attention_mask1" in batch else None
            labels1 = batch["labels1"] if "labels1" in batch else None
            
            input_ids2 = batch["input_ids2"]
            attention_mask2 = batch["attention_mask2"] if "attention_mask2" in batch else None
            labels2 = batch["labels2"] if "labels2" in batch else None
            
            # Get sample mode if available
            mode = batch.get("mode", ["auto"] * input_ids1.size(0))
            if isinstance(mode, list) and len(mode) > 0 and isinstance(mode[0], str):
                batch_mode = mode[0]
            else:
                batch_mode = "auto"
                
            # Set up exception handling to catch OOM errors
            try:
                # Forward pass with mixed precision if enabled
                if scaler:
                    with autocast(device_type='cuda'):
                        outputs = model(
                            input_ids1=input_ids1, 
                            input_ids2=input_ids2,
                            attention_mask1=attention_mask1, 
                            attention_mask2=attention_mask2,
                            labels1=labels1, 
                            labels2=labels2,
                            mode=batch_mode
                        )
                        loss = outputs["loss"]
                        if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                            loss = loss / accumulation_steps
                        else:
                            print("Warning: Invalid loss detected, skipping batch")
                            # Continue to next batch instead of using zero loss
                            continue
                    
                    # Scale loss and backprop
                    scaler.scale(loss).backward()
                    gradients_accumulated += 1
                    
                    # Only update when we've accumulated enough gradients
                    if gradients_accumulated >= accumulation_steps:
                        need_optimizer_step = True
                else:
                    # Standard full precision pass
                    outputs = model(
                        input_ids1=input_ids1, 
                        input_ids2=input_ids2,
                        attention_mask1=attention_mask1, 
                        attention_mask2=attention_mask2,
                        labels1=labels1, 
                        labels2=labels2,
                        mode=batch_mode
                    )
                    
                    loss = outputs["loss"]
                    if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                        loss = loss / accumulation_steps
                        loss.backward()
                        gradients_accumulated += 1
                    else:
                        print("Warning: Invalid loss detected, skipping batch")
                        continue
                    
                    # Only update when we've accumulated enough gradients
                    if gradients_accumulated >= accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                        optimizer.step()
                        optimizer.zero_grad()
                        gradients_accumulated = 0
                    
                # Ensure loss is on CPU for logging
                loss_item = loss.detach().cpu().item() * accumulation_steps
                total_train_loss += loss_item
                
                # Apply optimizer step if needed (for mixed precision)
                if need_optimizer_step and scaler:
                    # Unscale the gradients
                    scaler.unscale_(optimizer)
                    
                    # Clip gradients to prevent explosion
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                    
                    # Update weights and scaler
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Reset flags
                    need_optimizer_step = False
                    gradients_accumulated = 0
                
                # Count sample types
                detected_mode = outputs.get("mode", "unknown")
                if detected_mode == "single":
                    total_single_samples += input_ids1.size(0)
                elif detected_mode == "multi":
                    total_multi_samples += input_ids1.size(0)
                    
                # Log progress
                if i % 10 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Batch {i}/{len(train_dataloader)}, Loss: {loss_item:.4f}, LR: {current_lr:.6f}, Mode: {detected_mode}")
                    print(f"  Accumulated gradients: {gradients_accumulated}/{accumulation_steps}")
                    
                    # Log GPU memory for monitoring
                    if torch.cuda.is_available():
                        for d in range(torch.cuda.device_count()):
                            gpu_mem = torch.cuda.memory_allocated(d) / 1024**3  # GB
                            print(f"  GPU {d} Memory: {gpu_mem:.2f} GB")
                
                # Print sample outputs every print_interval batches
                if global_batch % print_interval == 0:
                    # Time stamp
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"\n=== MONITORING OUTPUT AT BATCH {global_batch} ({current_time}) ===")
                    print(f"Current loss: {loss_item:.4f}, Mode: {detected_mode}")
                    
                    # Get current predictions from the model
                    with torch.no_grad():
                        # Get the predictions from the logits
                        fused_logits1 = outputs.get("fused_logits1", None)
                        fused_logits2 = outputs.get("fused_logits2", None)
                        logits1 = outputs.get("logits1", None)
                        logits2 = outputs.get("logits2", None)
                        
                        # Process first sample in batch
                        sample_idx = 0
                        
                        # Get input prompts
                        input_prompt1 = tokenizer1.decode(input_ids1[sample_idx], skip_special_tokens=True)
                        input_prompt2 = tokenizer2.decode(input_ids2[sample_idx], skip_special_tokens=True)
                        
                        print(f"Input prompt for Model1: {input_prompt1[:50]}...")
                        print(f"Input prompt for Model2: {input_prompt2[:50]}...")
                        
                        # Generate output for the sample
                        sample_output = model.generate_dual(
                            input_ids1=input_ids1[sample_idx:sample_idx+1],
                            input_ids2=input_ids2[sample_idx:sample_idx+1],
                            tokenizer1=tokenizer1,
                            tokenizer2=tokenizer2,
                            max_length=50,  # Generate a short sample
                            temperature=0.7,
                            do_sample=True
                        )
                        
                        # Print outputs
                        print("\nModel1 Output:")
                        print(sample_output[0][0])
                        print("\nModel2 Output:")
                        print(sample_output[1][0])
                        
                        print(f"=== END MONITORING OUTPUT ===\n")
                            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: ran out of memory in batch {i}")
                    for d in range(torch.cuda.device_count()):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Skip this batch and move to the next one
                    optimizer.zero_grad()
                    need_optimizer_step = False
                    gradients_accumulated = 0
                    continue
                else:
                    raise e
                
        # Handle remaining gradient updates at the end of the epoch
        if gradients_accumulated > 0:
            if scaler:
                # Unscale the gradients
                scaler.unscale_(optimizer)
                
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                
                # Update weights and scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)
                optimizer.step()
            
            optimizer.zero_grad()
            
        # Calculate metrics
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time = time.time() - start_time
        
        # Log GPU memory usage if available
        gpu_mem_allocated = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_mem = torch.cuda.memory_allocated(i) / 1024**3  # GB
                print(f"GPU {i} Memory: {gpu_mem:.2f} GB")
                gpu_mem_allocated[i] = gpu_mem
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        val_single_samples = 0
        val_multi_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                input_ids1 = batch["input_ids1"]
                attention_mask1 = batch["attention_mask1"] if "attention_mask1" in batch else None
                labels1 = batch["labels1"] if "labels1" in batch else None
                
                input_ids2 = batch["input_ids2"]
                attention_mask2 = batch["attention_mask2"] if "attention_mask2" in batch else None
                labels2 = batch["labels2"] if "labels2" in batch else None
                
                # Get sample mode if available
                mode = batch.get("mode", ["auto"] * input_ids1.size(0))
                if isinstance(mode, list) and len(mode) > 0 and isinstance(mode[0], str):
                    batch_mode = mode[0]
                else:
                    batch_mode = "auto"
                
                try:
                    outputs = model(
                        input_ids1=input_ids1, 
                        input_ids2=input_ids2,
                        attention_mask1=attention_mask1, 
                        attention_mask2=attention_mask2,
                        labels1=labels1, 
                        labels2=labels2,
                        mode=batch_mode
                    )
                    
                    loss = outputs["loss"]
                    if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                        total_val_loss += loss.detach().cpu().item()
                    
                    # Count sample types based on detected mode
                    detected_mode = outputs.get("mode", "unknown")
                    if detected_mode == "single":
                        val_single_samples += input_ids1.size(0)
                    elif detected_mode == "multi":
                        val_multi_samples += input_ids1.size(0)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: ran out of memory during validation")
                        for d in range(torch.cuda.device_count()):
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else float('inf')
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s, GPU Memory: {gpu_mem_allocated}")
        print(f"Train samples - Single: {total_single_samples}, Multi: {total_multi_samples}")
        print(f"Val samples - Single: {val_single_samples}, Multi: {val_multi_samples}")
        
        # Generate and print sample outputs at the end of each epoch
        print("\n=== END OF EPOCH SAMPLE OUTPUTS ===")
        
        # Get a sample from validation data
        sample_batch = next(iter(val_dataloader))
        sample_idx = 0
        
        with torch.no_grad():
            # Generate output for the sample
            sample_output = model.generate_dual(
                input_ids1=sample_batch["input_ids1"][sample_idx:sample_idx+1],
                input_ids2=sample_batch["input_ids2"][sample_idx:sample_idx+1],
                tokenizer1=tokenizer1,
                tokenizer2=tokenizer2,
                max_length=100,  # Generate a reasonable sample
                temperature=0.7,
                do_sample=True
            )
            
            # Print outputs
            input_prompt1 = tokenizer1.decode(sample_batch["input_ids1"][sample_idx], skip_special_tokens=True)
            input_prompt2 = tokenizer2.decode(sample_batch["input_ids2"][sample_idx], skip_special_tokens=True)
            
            print(f"Input prompt for Model1: {input_prompt1}")
            print(f"Model1 Output:")
            print(sample_output[0][0])
            
            print(f"Input prompt for Model2: {input_prompt2}")
            print(f"Model2 Output:")
            print(sample_output[1][0])
        
        print("=== END OF SAMPLE OUTPUTS ===\n")
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            try:
                # Save model state dict only for the trainable components - UPDATED for dual fused heads
                state_dict = {
                    'fusion_layer': model.fusion_layer.state_dict(),
                    'proj1': model.proj1.state_dict(),
                    'proj2': model.proj2.state_dict(),
                    'fused_lm_head1': model.fused_lm_head1.state_dict(),  # Updated for dual heads
                    'fused_lm_head2': model.fused_lm_head2.state_dict(),  # Updated for dual heads
                    'lm_head1': model.lm_head1.state_dict(),
                    'lm_head2': model.lm_head2.state_dict(),
                    'task_classifier': model.task_classifier.state_dict()
                }
                torch.save(state_dict, os.path.join(output_dir, "dual_model_best.pt"))
                print(f"Saved new best model with validation loss: {best_val_loss:.4f}")
            except Exception as e:
                print(f"Error saving best model: {e}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Save checkpoint for this epoch - UPDATED for dual fused heads
        try:
            checkpoint = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'fusion_layer': model.fusion_layer.state_dict(),
                'proj1': model.proj1.state_dict(),
                'proj2': model.proj2.state_dict(),
                'fused_lm_head1': model.fused_lm_head1.state_dict(),  # Updated for dual heads
                'fused_lm_head2': model.fused_lm_head2.state_dict(),  # Updated for dual heads
                'lm_head1': model.lm_head1.state_dict(),
                'lm_head2': model.lm_head2.state_dict(),
                'task_classifier': model.task_classifier.state_dict()
            }
            torch.save(checkpoint, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
    
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    return best_val_loss

# ----- Create the Complete Model -----

def create_dual_decoder_model(config, tokenizer1, tokenizer2):
    # Check GPU availability
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"Available GPUs: {gpu_count}")
    
    # Setup device map for multi-GPU use with memory balancing
    device_map = {
        'model1': 'cuda:0',       # First base model on GPU 0
        'model2': 'cuda:0',       # Second base model on GPU 0
        'fusion': 'cuda:1',       # Fusion layer on GPU 1
        'extra': 'cuda:0'         # Extra computations on GPU 0 to balance memory
    }
    
    print("Using memory-balanced device mapping:")
    print(f" - Base models will be on: {device_map['model1']}")
    print(f" - Fusion layer will be on: {device_map['fusion']}")
    print(f" - Model2 heads will be on: {device_map['extra']} (to balance memory)")
    
    # Force torch.cuda to use the specified devices when loading models
    print("Loading base models with explicit device placement...")
    
    # Load model1 onto cuda:0
    torch.cuda.set_device(int(device_map['model1'].split(':')[1]))
    model1 = AutoModel.from_pretrained(config["model_dir1"])
    model1.to(device_map['model1'])
    print(f"Model1 loaded - device: {next(model1.parameters()).device}")
    
    # Load model2 onto cuda:0
    torch.cuda.set_device(int(device_map['model2'].split(':')[1]))
    model2 = AutoModel.from_pretrained(config["model_dir2"])
    model2.to(device_map['model2'])
    print(f"Model2 loaded - device: {next(model2.parameters()).device}")
    
    # Create the dual decoder model with the balanced device mapping
    print("Creating dual decoder model with memory-balanced architecture...")
    dual_model = DualDecoderModel(
        model1, 
        model2,
        tokenizer1,
        tokenizer2,
        config["fusion_output_dim"],
        freeze_base_models=config.get("freeze_base_models", True),
        device_map=device_map
    )
    
    # Print GPU memory usage for verification
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_mem = torch.cuda.memory_allocated(i) / 1024**3
            print(f"GPU {i} memory usage after model loading: {gpu_mem:.2f} GB")
    
    return dual_model

# ----- Example Queries -----

def run_example_queries(model, tokenizer1, tokenizer2, device_map=None):
    """
    Run example queries to test the model
    
    Args:
        model: Trained DualDecoderModel
        tokenizer1: Tokenizer for model1
        tokenizer2: Tokenizer for model2
        device_map: Device mapping configuration
    """
    # Move model to appropriate devices for inference
    if device_map:
        # Updated to move both fused heads to fusion device
        model.model1.to(device_map['model1'])
        model.model2.to(device_map['model2'])
        model.fusion_layer.to(device_map['fusion'])
        model.fused_lm_head1.to(device_map['fusion'])  # Added
        model.fused_lm_head2.to(device_map['fusion'])  # Added
        model.lm_head1.to(device_map['fusion'])
        model.lm_head2.to(device_map['fusion'])
    else:
        primary_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(primary_device)
    
    model.eval()
    
    # Example 1: Same prompt to both models
    prompt = "Write a short story about a robot learning to feel emotions."
    print("\n=== Example 1: Same Prompt ===")
    print(f"Prompt: {prompt}")
    
    # Tokenize with both tokenizers
    inputs1 = tokenizer1(prompt, return_tensors="pt")
    inputs2 = tokenizer2(prompt, return_tensors="pt")  # Using same prompt but correct tokenizer
    
    # Generate responses
    response1, response2 = model.generate_dual(
        inputs1["input_ids"], 
        inputs2["input_ids"], 
        tokenizer1, 
        tokenizer2,
        max_length=50
    )
    
    print("\nModel 1 (Qwen) Response:")
    print(response1[0])
    print("\nModel 2 (Llama) Response:")
    print(response2[0])
    
    # Example 2: Different prompts to each model
    prompt1 = "Explain the theory of relativity in simple terms."
    prompt2 = "Write a poem about the beauty of mathematics."
    
    print("\n=== Example 2: Different Prompts ===")
    print(f"Prompt 1: {prompt1}")
    print(f"Prompt 2: {prompt2}")
    
    # Tokenize with respective tokenizers
    inputs1 = tokenizer1(prompt1, return_tensors="pt")
    inputs2 = tokenizer2(prompt2, return_tensors="pt")
    
    # Generate responses
    response1, response2 = model.generate_dual(
        inputs1["input_ids"], 
        inputs2["input_ids"], 
        tokenizer1, 
        tokenizer2,
        max_length=200
    )
    
    print("\nModel 1 (Qwen) Response (to prompt 1):")
    print(response1[0])
    print("\nModel 2 (Llama) Response (to prompt 2):")
    print(response2[0])
    
    
def debug_tensor_devices(model, sample_batch):
    """
    Debug function to trace tensor movements across devices during a forward pass
    
    Args:
        model: The DualDecoderModel
        sample_batch: A sample batch to process
    """
    print("\n=== DEBUGGING TENSOR DEVICES ===")
    
    # Get input tensors
    input_ids1 = sample_batch["input_ids1"]
    attention_mask1 = sample_batch["attention_mask1"] if "attention_mask1" in sample_batch else None
    input_ids2 = sample_batch["input_ids2"] 
    attention_mask2 = sample_batch["attention_mask2"] if "attention_mask2" in sample_batch else None
    
    print(f"Initial input devices:")
    print(f"  input_ids1: {input_ids1.device}")
    if attention_mask1 is not None:
        print(f"  attention_mask1: {attention_mask1.device}")
    print(f"  input_ids2: {input_ids2.device}")
    if attention_mask2 is not None:
        print(f"  attention_mask2: {attention_mask2.device}")
    
    # Check model device placement - Updated to check both fused heads
    print("\nModel component devices:")
    print(f"  model1: {next(model.model1.parameters()).device}")
    print(f"  model2: {next(model.model2.parameters()).device}")
    print(f"  fusion_layer: {next(model.fusion_layer.parameters()).device}")
    print(f"  fused_lm_head1: {model.fused_lm_head1.weight.device}")  # Added
    print(f"  fused_lm_head2: {model.fused_lm_head2.weight.device}")  # Added
    print(f"  lm_head1: {model.lm_head1.weight.device}")
    print(f"  lm_head2: {model.lm_head2.weight.device}")
    
    # Run model with hooks to track tensor devices
    device_tracking = []
    
    def add_hook(module, input, output, name):
        if isinstance(output, tuple):
            output = output[0]  # Take first tensor if output is a tuple
        
        if hasattr(output, 'device'):
            device_tracking.append({
                'stage': name,
                'device': str(output.device),
                'shape': tuple(output.shape) if hasattr(output, 'shape') else None
            })
    
    # Register hooks - Updated to track both fused heads
    hooks = []
    hooks.append(model.model1.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "model1_output")))
    hooks.append(model.model2.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "model2_output")))
    hooks.append(model.fusion_layer.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "fusion_output")))
    hooks.append(model.fused_lm_head1.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "fused_lm_head1_output")))  # Added
    hooks.append(model.fused_lm_head2.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "fused_lm_head2_output")))  # Added
    hooks.append(model.lm_head1.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "lm_head1_output")))
    hooks.append(model.lm_head2.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "lm_head2_output")))
    
    # Run a forward pass with tracing
    try:
        print("\nRunning forward pass with tracing...")
        with torch.no_grad():
            outputs = model(
                input_ids1=input_ids1,
                input_ids2=input_ids2,
                attention_mask1=attention_mask1,
                attention_mask2=attention_mask2,
                mode="auto"
            )
        
        # Print tensor device trace
        print("\nTensor device trace:")
        for item in device_tracking:
            print(f"  {item['stage']}: {item['device']}, shape: {item['shape']}")
        
        # Print output devices
        print("\nOutput devices:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.device}, shape: {value.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()
    
    print("\n=== GPU MEMORY USAGE ===")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
    
    print("\n=== DEBUGGING COMPLETE ===")


def memory_profiling(model, sample_batch):
    """
    Profile memory usage during a forward and backward pass
    
    Args:
        model: The DualDecoderModel
        sample_batch: A sample batch to process
    """
    print("\n=== MEMORY PROFILING ===")
    
    # Get initial memory
    initial_mem = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            initial_mem[i] = torch.cuda.memory_allocated(i) / 1024**3
    
    # Extract batch
    input_ids1 = sample_batch["input_ids1"]
    attention_mask1 = sample_batch["attention_mask1"] if "attention_mask1" in sample_batch else None
    labels1 = sample_batch["labels1"] if "labels1" in sample_batch else None
    input_ids2 = sample_batch["input_ids2"]
    attention_mask2 = sample_batch["attention_mask2"] if "attention_mask2" in sample_batch else None
    labels2 = sample_batch["labels2"] if "labels2" in sample_batch else None
    
    # Step 1: Forward pass
    print("\nStep 1: Forward pass")
    model.train()  # Set to training mode
    
    try:
        outputs = model(
            input_ids1=input_ids1,
            input_ids2=input_ids2,
            attention_mask1=attention_mask1,
            attention_mask2=attention_mask2,
            labels1=labels1,
            labels2=labels2,
            mode="auto"
        )
        
        # Print out the keys in outputs to see what's available
        print(f"Output keys: {list(outputs.keys())}")  # Added to show both fused_logits1 and fused_logits2
        
        # Check memory after forward pass
        forward_mem = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                forward_mem[i] = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} - After forward: {forward_mem[i]:.2f} GB (Δ: {forward_mem[i] - initial_mem[i]:.2f} GB)")
        
        # Step 2: Backward pass
        print("\nStep 2: Backward pass")
        loss = outputs["loss"]
        loss.backward()
        
        # Check memory after backward pass
        backward_mem = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                backward_mem[i] = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} - After backward: {backward_mem[i]:.2f} GB (Δ from forward: {backward_mem[i] - forward_mem[i]:.2f} GB)")
        
        # Step 3: Optimizer step
        print("\nStep 3: Memory after gradient zeroing")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        
        # Check memory after zero_grad
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                zero_grad_mem = torch.cuda.memory_allocated(i) / 1024**3
                print(f"GPU {i} - After zero_grad: {zero_grad_mem:.2f} GB (Δ from backward: {zero_grad_mem - backward_mem[i]:.2f} GB)")
    
    except Exception as e:
        print(f"Error during profiling: {e}")
    
    print("\n=== PROFILING COMPLETE ===")
