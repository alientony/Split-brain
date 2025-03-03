import os
import time
import torch
from torch.utils.data import DataLoader

# Import our modules
from dual_mode_data_preparation import (
    prepare_dataset_for_training_chunked,
    convert_single_to_multi_format,
    create_mixed_dataset
)
from dual_mode_model_training import (
    create_dual_decoder_model, 
    train_dual_model,
    run_example_queries
)

def main():
    # Configuration settings
    config = {
        # Model paths
        "model_dir1": "./DeepSeek-R1-Distill-Qwen-1.5B",
        "model_dir2": "./Llama-3.2-1B",
        "dataset_path": "data.jsonl",
        "output_dir": "./model_outputs",
        
        # Model configuration
        "fusion_output_dim": 2048,  # Larger fusion layer for cross-attention
        "freeze_base_models": True,
        "max_length": 1600,
        
        # Training configuration
        "train_batch_size": 1,
        "eval_batch_size": 4,
        "learning_rate": 5e-5,
        "epochs": 3,
        "patience": 2,
        "accumulation_steps": 16,
        "train_val_split": 0.8,
        "seed": 42,
        "use_multi_gpu": True,
        
        # Dataset loading configuration
        "data_workers": 12,             # Number of parallel workers
        "data_batch_size": 1000,        # Process this many samples in each parallel batch
        "use_threads": True,            # Use threads instead of processes
        "dataset_cache_dir": "./dataset_cache",  # Caching directory
        "use_cached_dataset": True,     # Whether to use/save cached preprocessed dataset

        
        # Data preparation (optional)
        "prepare_mixed_dataset": False,  # Whether to create a mixed dataset
        "single_format_file": "single_data.jsonl",  # Path to single-task format file
        "multi_format_file": "multi_data.jsonl",    # Path to multi-task format file
        "mixed_output_file": "mixed_data.jsonl",    # Path to save the mixed dataset
        "single_task_ratio": 0.5,        # Ratio of single-task samples in mixed dataset
        "processing_chunk_size": 100000,  # Number of examples to process at once
        "data_batch_size": 1000,         # Size of batches for parallel processing

        "create_multitask_samples": True,  # Enable multi-task conversion
        "multitask_ratio": 0.1,           # 30% of samples become multi-task        
    }
    
    # Set random seed for reproducibility
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    
    # Create output directory if not exists
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Optional: Create a mixed dataset if requested
    if config["prepare_mixed_dataset"]:
        print("Preparing mixed dataset...")
        
        # Step 1: Convert original data to multi-task format if needed
        convert_single_to_multi_format(
            config["single_format_file"],
            config["multi_format_file"],
            mode="shuffle"  # Options: duplicate, alternate, shuffle
        )
        
        # Step 2: Create mixed dataset with both formats
        create_mixed_dataset(
            config["single_format_file"],
            config["multi_format_file"],
            config["mixed_output_file"],
            ratio=config["single_task_ratio"]
        )
        
        # Update the dataset path
        config["dataset_path"] = config["mixed_output_file"]
    
    # Prepare dataset for training
    print("Preparing dataset...")
    train_dataloader, val_dataloader, tokenizer1, tokenizer2 = prepare_dataset_for_training_chunked(config)
    
    # Create the dual decoder model
    dual_model = create_dual_decoder_model(config, tokenizer1, tokenizer2)
    
    # Set up optimizer - we only optimize parameters that require gradients
    trainable_params = [p for p in dual_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"])
    
    # Train the model
    print("Starting training...")
    train_dual_model(
        model=dual_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        epochs=config["epochs"],
        patience=config["patience"],
        accumulation_steps=config["accumulation_steps"],
        fp16=True,
        output_dir=config["output_dir"],
        device_map=dual_model.device_map
    )
    
    # Save the final model
    print("Saving model...")
    # Move all components to CPU for saving to avoid GPU memory issues
    dual_model.to("cpu")
    final_model_path = os.path.join(config["output_dir"], "dual_model_final.pt")
    torch.save(dual_model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Run example queries
    print("\nRunning example queries...")
    # Get device map for inference
    device_map = None
    if torch.cuda.device_count() > 1 and config["use_multi_gpu"]:
        device_map = {
            'model1': 'cuda:1',
            'model2': 'cuda:0',
            'fusion': 'cuda:0'
        }
    
    run_example_queries(dual_model, tokenizer1, tokenizer2, device_map)
    print("Example queries complete!")

if __name__ == "__main__":
    main()
