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
        super().__init__()
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
        self.scale = fusion_output_dim ** 0.5  # More numerically stable

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
        Enhanced DualDecoderModel with multi-task capabilities and stronger fusion.
        
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
        
        # Language model heads for various outputs
        # 1. Main fused output for model1 vocabulary
        self.fused_lm_head = nn.Linear(fusion_output_dim, self.vocab_size1, bias=False)
        
        # 2. Model1-specific head using its own attention + fusion 
        self.lm_head1 = nn.Linear(fusion_output_dim, self.vocab_size1, bias=False)
        
        # 3. Model2-specific head using its own attention + fusion
        self.lm_head2 = nn.Linear(fusion_output_dim, self.vocab_size2, bias=False)
        
        # Task type classifier - determine if inputs are the same or different
        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_size1 + self.hidden_size2, fusion_output_dim),
            nn.GELU(),
            nn.Linear(fusion_output_dim, 1),
            nn.Sigmoid()
        )
        
        # Move models to respective devices after initialization
        self.model1.to(self.device1)
        self.model2.to(self.device2)
        self.fusion_layer.to(self.fusion_device)
        self.proj1.to(self.fusion_device)
        self.proj2.to(self.fusion_device)
        self.fused_lm_head.to(self.fusion_device)
        self.lm_head1.to(self.fusion_device)
        self.lm_head2.to(self.fusion_device)
        self.task_classifier.to(self.fusion_device)
        
        if freeze_base_models:
            for param in self.model1.parameters():
                param.requires_grad = False
            for param in self.model2.parameters():
                param.requires_grad = False

    def forward(self, input_ids1, input_ids2, attention_mask1=None, attention_mask2=None, 
                labels1=None, labels2=None, mode="auto", return_details=False):
        """
        Forward pass supporting both single-prompt and dual-prompt modes
        
        Args:
            input_ids1: Input tokens for model1
            input_ids2: Input tokens for model2
            attention_mask1: Attention mask for model1
            attention_mask2: Attention mask for model2
            labels1: Target labels for model1
            labels2: Target labels for model2
            mode: Processing mode - "auto", "single" (same prompt) or "multi" (different prompts)
            return_details: Whether to return detailed fusion information
        
        Returns:
            Dictionary with loss and logits
        """
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
        
        # Move hidden states to fusion device for further processing
        hidden1 = hidden1.to(self.fusion_device)
        hidden2 = hidden2.to(self.fusion_device)
        
        # Project embeddings to fusion dimension if needed
        if self.hidden_size1 != self.fusion_layer.fusion_output_dim:
            hidden1_proj = self.proj1(hidden1)
        else:
            hidden1_proj = hidden1
            
        if self.hidden_size2 != self.fusion_layer.fusion_output_dim:
            hidden2_proj = self.proj2(hidden2)
        else:
            hidden2_proj = hidden2
        
        # Determine task type (are inputs the same or different?)
        if mode == "auto":
            cls1 = hidden1[:, 0, :]  # First token representation from model1
            cls2 = hidden2[:, 0, :]  # First token representation from model2
            task_input = torch.cat([cls1, cls2], dim=-1)
            task_prob = self.task_classifier(task_input)
            # If prob > 0.5, likely multi-task mode, else single-task
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
        
        # Generate logits using different heads
        # 1. Shared fused representation (primary output for single-task mode)
        fused_logits = self.fused_lm_head(fused)
        
        # 2. Model1-specific output using cross-attended representation
        logits1 = self.lm_head1(out1)
        
        # 3. Model2-specific output using cross-attended representation
        logits2 = self.lm_head2(out2)
        
        # Calculate losses if labels are provided
        loss = None
        loss1 = None
        loss2 = None
        total_loss = None
        
        if labels1 is not None:
            labels1 = labels1.to(self.fusion_device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Prepare shifted logits and labels for autoregressive training
            # Handle case where sequence length could be 1
            if fused_logits.size(1) > 1 and labels1.size(1) > 1:
                shift_fused_logits = fused_logits[..., :-1, :].contiguous()
                shift_logits1 = logits1[..., :-1, :].contiguous()
                shift_labels1 = labels1[..., 1:].contiguous()
                
                # Calculate losses
                if detected_mode == "single":
                    # In single mode, the main fused output should predict model1's labels
                    loss = loss_fct(shift_fused_logits.view(-1, shift_fused_logits.size(-1)), 
                                    shift_labels1.view(-1))
                
                # Model1-specific loss
                loss1 = loss_fct(shift_logits1.view(-1, shift_logits1.size(-1)), 
                                 shift_labels1.view(-1))
            else:
                # Handle single token cases or empty sequences
                if detected_mode == "single":
                    loss = torch.tensor(0.0, requires_grad=True, device=self.fusion_device)
                loss1 = torch.tensor(0.0, requires_grad=True, device=self.fusion_device) 
        
        if labels2 is not None:
            labels2 = labels2.to(self.fusion_device)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            
            # Prepare shifted logits and labels
            if logits2.size(1) > 1 and labels2.size(1) > 1:
                shift_logits2 = logits2[..., :-1, :].contiguous()
                shift_labels2 = labels2[..., 1:].contiguous()
                
                # Model2-specific loss
                loss2 = loss_fct(shift_logits2.view(-1, shift_logits2.size(-1)), 
                                 shift_labels2.view(-1))
            else:
                loss2 = torch.tensor(0.0, requires_grad=True, device=self.fusion_device)
        
        # Determine final combined loss based on mode
        if loss1 is not None and loss2 is not None:
            if detected_mode == "single":
                # In single-task mode, the fused representation is more important
                if loss is not None:
                    total_loss = loss + 0.3 * loss1 + 0.3 * loss2
                else:
                    total_loss = 0.4 * loss1 + 0.6 * loss2
            else:
                # In multi-task mode, the separate decoders are more important
                if loss is not None:
                    total_loss = 0.2 * loss + 0.4 * loss1 + 0.4 * loss2
                else:
                    total_loss = 0.5 * loss1 + 0.5 * loss2
        elif loss1 is not None:
            total_loss = loss1
        elif loss2 is not None:
            total_loss = loss2
        elif loss is not None:
            total_loss = loss
        
        if return_details:
            return {
                "loss": total_loss,
                "fused_logits": fused_logits,
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
                "fused_logits": fused_logits,
                "logits1": logits1,
                "logits2": logits2,
                "mode": detected_mode
            }

# ----- Training Function -----

def train_dual_model(model, train_dataloader, val_dataloader, optimizer, 
                     epochs=3, patience=2, accumulation_steps=8, fp16=True,
                     output_dir="./outputs", device_map=None):
    """
    Train the dual decoder model with both single-task and multi-task samples
    
    Args:
        model: The DualDecoderModel to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for training
        epochs: Number of training epochs
        patience: Early stopping patience
        accumulation_steps: Gradient accumulation steps
        fp16: Whether to use mixed precision training
        output_dir: Directory to save model checkpoints
        device_map: Dictionary mapping model components to devices
    
    Returns:
        best_val_loss: Best validation loss achieved
    """
    import torch  # Add this line to ensure torch is available    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device information for reporting
    device1 = device_map.get('model1', 'cuda:1') if device_map else 'cuda:1'
    device2 = device_map.get('model2', 'cuda:0' if torch.cuda.device_count() > 1 else 'cuda:0') if device_map else 'cuda:0'
    fusion_device = device_map.get('fusion', 'cuda:1') if device_map else 'cuda:0'
    
    print(f"Training with devices - Model1: {device1}, Model2: {device2}, Fusion: {fusion_device}")
    
    # Initialize mixed precision training if requested and available
    scaler = None
    if fp16 and torch.cuda.is_available():
        # Use version-compatible initialization
        import torch.cuda.amp
        scaler = torch.cuda.amp.GradScaler()
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
        
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")):
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
                # If mode is a list of strings, pick the first one
                batch_mode = mode[0]
            else:
                batch_mode = "auto"
                
            # Set up exception handling to catch OOM errors
            try:
                # Forward pass with mixed precision if enabled
                if scaler:
                    with torch.cuda.amp.autocast():
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
                        loss = loss / accumulation_steps
                    
                    # Scale loss and backprop
                    scaler.scale(loss).backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)
                        
                        # Clip gradients to prevent explosion
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # Update weights and scaler
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
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
                    loss = loss / accumulation_steps
                    loss.backward()
                    
                    if (i + 1) % accumulation_steps == 0:
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                    
                # Ensure loss is on CPU for logging
                loss_item = loss.detach().cpu().item() * accumulation_steps
                total_train_loss += loss_item
                
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
                    
                    # Log GPU memory for monitoring
                    if torch.cuda.is_available():
                        for d in range(torch.cuda.device_count()):
                            gpu_mem = torch.cuda.memory_allocated(d) / 1024**3  # GB
                            print(f"  GPU {d} Memory: {gpu_mem:.2f} GB")
                            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"WARNING: ran out of memory in batch {i}")
                    for d in range(torch.cuda.device_count()):
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    # Skip this batch and move to the next one
                    optimizer.zero_grad()
                    continue
                else:
                    raise e
                
        # Handle remaining gradient updates
        if len(train_dataloader) % accumulation_steps != 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            try:
                # Save model state dict only for the trainable components
                state_dict = {
                    'fusion_layer': model.fusion_layer.state_dict(),
                    'proj1': model.proj1.state_dict(),
                    'proj2': model.proj2.state_dict(),
                    'fused_lm_head': model.fused_lm_head.state_dict(),
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
        
        # Save checkpoint for this epoch
        try:
            checkpoint = {
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'fusion_layer': model.fusion_layer.state_dict(),
                'proj1': model.proj1.state_dict(),
                'proj2': model.proj2.state_dict(),
                'fused_lm_head': model.fused_lm_head.state_dict(),
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
    
    # Setup device map for single-GPU base models and separate fusion training
    # Both Model1 and Model2 will be on cuda:0 while fusion and training components will use cuda:1.
    device_map = {
        'model1': 'cuda:0',
        'model2': 'cuda:0',
        'fusion': 'cuda:1'
    }
    
    print("Using device mapping:")
    print(f" - Model1 will be on: {device_map['model1']}")
    print(f" - Model2 will be on: {device_map['model2']}")
    print(f" - Fusion components will be on: {device_map['fusion']}")
    
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
    
    # Create the dual decoder model with the new device mapping
    print("Creating dual decoder model with updated device mapping...")
    dual_model = DualDecoderModel(
        model1, 
        model2,
        tokenizer1,
        tokenizer2,
        config["fusion_output_dim"],
        freeze_base_models=config.get("freeze_base_models", True),
        device_map=device_map
    )
    
    # Verify model device placement
    print("\nVerifying model device placement:")
    print(f"Model1 device: {next(dual_model.model1.parameters()).device}")
    print(f"Model2 device: {next(dual_model.model2.parameters()).device}")
    print(f"Fusion layer device: {next(dual_model.fusion_layer.parameters()).device}")
    print(f"Projection layers devices: {next(dual_model.proj1.parameters()).device}, {next(dual_model.proj2.parameters()).device}")
    print(f"Language model heads devices: {next(dual_model.lm_head1.parameters()).device}, {next(dual_model.lm_head2.parameters()).device}")
    
    # Optionally print GPU memory usage for verification
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
        model.model1.to(device_map['model1'])
        model.model2.to(device_map['model2'])
        model.fusion_layer.to(device_map['fusion'])
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
    inputs2 = tokenizer1(prompt, return_tensors="pt")  # Using same tokenizer outputs
    
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
        max_length=50
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
    
    # Check model device placement
    print("\nModel component devices:")
    print(f"  model1: {next(model.model1.parameters()).device}")
    print(f"  model2: {next(model.model2.parameters()).device}")
    print(f"  fusion_layer: {next(model.fusion_layer.parameters()).device}")
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
    
    # Register hooks
    hooks = []
    hooks.append(model.model1.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "model1_output")))
    hooks.append(model.model2.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "model2_output")))
    hooks.append(model.fusion_layer.register_forward_hook(
        lambda module, input, output: add_hook(module, input, output, "fusion_output")))
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
