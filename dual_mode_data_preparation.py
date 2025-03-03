import json
import os
import time
import pickle
import hashlib
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

# ----- Data Sample Processing Functions -----

def process_single_task_sample(line, tokenizer1, tokenizer2, max_length=512):
    """
    Process a single sample where the same prompt goes to both models
    Format: {"prompt": "...", "response": "..."}
    """
    data = json.loads(line)
    prompt = data["prompt"]
    response = data["response"]
    
    # Tokenize with first tokenizer
    prompt_tokens1 = tokenizer1(prompt, return_tensors="pt")
    prompt_len1 = prompt_tokens1["input_ids"].size(1)
    
    full_text1 = prompt + "\n\n" + response
    tokenized1 = tokenizer1(
        full_text1, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # Tokenize with second tokenizer
    prompt_tokens2 = tokenizer2(prompt, return_tensors="pt")
    prompt_len2 = prompt_tokens2["input_ids"].size(1)
    
    full_text2 = prompt + "\n\n" + response
    tokenized2 = tokenizer2(
        full_text2, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # Remove batch dimension
    sample1 = {k: v.squeeze(0) for k, v in tokenized1.items()}
    sample2 = {k: v.squeeze(0) for k, v in tokenized2.items()}
    
    # Prepare labels for both models
    sample1["labels"] = sample1["input_ids"].clone()
    sample2["labels"] = sample2["input_ids"].clone()
    
    # Mask prompt tokens in labels
    prompt_len1 = min(prompt_len1, max_length)
    prompt_len2 = min(prompt_len2, max_length)
    
    sample1["labels"][:prompt_len1] = -100
    sample2["labels"][:prompt_len2] = -100
    
    # Combined sample with metadata
    combined_sample = {
        "input_ids1": sample1["input_ids"],
        "attention_mask1": sample1["attention_mask"],
        "labels1": sample1["labels"],
        "input_ids2": sample2["input_ids"],
        "attention_mask2": sample2["attention_mask"],
        "labels2": sample2["labels"],
        "mode": "single",  # Both models got the same prompt
        "prompt1": prompt,
        "prompt2": prompt,
        "response1": response,
        "response2": response
    }
    
    return combined_sample

def process_multi_task_sample(line, tokenizer1, tokenizer2, max_length=512):
    """
    Process a dual sample where each model gets a different prompt
    Format: {"prompt1": "...", "response1": "...", "prompt2": "...", "response2": "..."}
    """
    data = json.loads(line)
    prompt1 = data["prompt1"] 
    response1 = data["response1"]
    prompt2 = data["prompt2"]
    response2 = data["response2"]
    
    # Tokenize for first model
    prompt_tokens1 = tokenizer1(prompt1, return_tensors="pt")
    prompt_len1 = prompt_tokens1["input_ids"].size(1)
    
    full_text1 = prompt1 + "\n\n" + response1
    tokenized1 = tokenizer1(
        full_text1, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # Tokenize for second model
    prompt_tokens2 = tokenizer2(prompt2, return_tensors="pt")
    prompt_len2 = prompt_tokens2["input_ids"].size(1)
    
    full_text2 = prompt2 + "\n\n" + response2
    tokenized2 = tokenizer2(
        full_text2, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    
    # Remove batch dimension
    sample1 = {k: v.squeeze(0) for k, v in tokenized1.items()}
    sample2 = {k: v.squeeze(0) for k, v in tokenized2.items()}
    
    # Prepare labels for both models
    sample1["labels"] = sample1["input_ids"].clone()
    sample2["labels"] = sample2["input_ids"].clone()
    
    # Mask prompt tokens in labels
    prompt_len1 = min(prompt_len1, max_length)
    prompt_len2 = min(prompt_len2, max_length)
    
    sample1["labels"][:prompt_len1] = -100
    sample2["labels"][:prompt_len2] = -100
    
    # Combined sample with metadata
    combined_sample = {
        "input_ids1": sample1["input_ids"],
        "attention_mask1": sample1["attention_mask"],
        "labels1": sample1["labels"],
        "input_ids2": sample2["input_ids"],
        "attention_mask2": sample2["attention_mask"],
        "labels2": sample2["labels"],
        "mode": "multi",  # Each model got a different prompt
        "prompt1": prompt1,
        "prompt2": prompt2,
        "response1": response1,
        "response2": response2
    }
    
    return combined_sample

def detect_sample_format(line):
    """Detect if a sample is single-task or multi-task format"""
    data = json.loads(line)
    if "prompt" in data and "response" in data:
        return "single"
    elif "prompt1" in data and "prompt2" in data and "response1" in data and "response2" in data:
        return "multi"
    else:
        # If format is unclear, default to single if we can
        if "prompt" in data and "response" in data:
            return "single"
        else:
            raise ValueError(f"Unknown sample format: {data.keys()}")

def process_combined_sample(line, tokenizer1, tokenizer2, max_length=512):
    """Process a sample, determining its format first"""
    format_type = detect_sample_format(line)
    
    if format_type == "single":
        return process_single_task_sample(line, tokenizer1, tokenizer2, max_length)
    else:
        return process_multi_task_sample(line, tokenizer1, tokenizer2, max_length)

def process_batch(batch_lines, tokenizer1_path, tokenizer2_path, max_length):
    """
    Process a batch of lines with fresh tokenizer instances to avoid threading issues.
    
    Args:
        batch_lines: List of lines to process
        tokenizer1_path: Path to first tokenizer
        tokenizer2_path: Path to second tokenizer
        max_length: Maximum sequence length
        
    Returns:
        List of processed samples
    """
    # Create fresh tokenizer instances for this batch/thread
    try:
        # Load tokenizers in the worker thread
        tokenizer1 = AutoTokenizer.from_pretrained(tokenizer1_path)
        tokenizer2 = AutoTokenizer.from_pretrained(tokenizer2_path)
        
        # Handle special tokens
        if tokenizer1.pad_token is None:
            tokenizer1.pad_token = tokenizer1.eos_token
        
        if tokenizer2.pad_token is None:
            tokenizer2.pad_token = tokenizer2.eos_token
        
        results = []
        for line in batch_lines:
            try:
                # Detect and process sample format
                format_type = detect_sample_format(line)
                
                if format_type == "single":
                    sample = process_single_task_sample(line, tokenizer1, tokenizer2, max_length)
                else:
                    sample = process_multi_task_sample(line, tokenizer1, tokenizer2, max_length)
                    
                results.append(sample)
            except Exception as e:
                print(f"Error processing line: {e}")
                print(f"Line content: {line[:100]}...")
        
        return results
    except Exception as e:
        print(f"Error in batch processing: {e}")
        import traceback
        traceback.print_exc()
        return []

def create_dynamic_mixed_dataset(samples, multi_task_ratio=0.3, seed=42):
    """
    Dynamically convert some single-task samples to multi-task samples.
    
    Args:
        samples: List of single-task samples
        multi_task_ratio: Ratio of samples to convert to multi-task format (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        List of mixed single-task and multi-task samples
    """
    import random
    random.seed(seed)
    
    # Determine how many samples to convert
    total_samples = len(samples)
    num_to_convert = int(total_samples * multi_task_ratio)
    
    # Ensure it's an even number (we need pairs)
    if num_to_convert % 2 != 0:
        num_to_convert -= 1
    
    # Make a copy of samples
    all_samples = samples.copy()
    random.shuffle(all_samples)
    
    # Split into samples to convert and samples to keep as-is
    to_convert = all_samples[:num_to_convert]
    to_keep = all_samples[num_to_convert:]
    
    # Create multi-task samples by pairing
    multi_task_samples = []
    for i in range(0, len(to_convert), 2):
        if i+1 >= len(to_convert):
            break
            
        sample1 = to_convert[i]
        sample2 = to_convert[i+1]
        
        # Create a multi-task sample
        multi_sample = {
            "input_ids1": sample1["input_ids1"],
            "attention_mask1": sample1["attention_mask1"],
            "labels1": sample1["labels1"],
            "input_ids2": sample2["input_ids2"],
            "attention_mask2": sample2["attention_mask2"],
            "labels2": sample2["labels2"],
            "mode": "multi",
            "prompt1": sample1["prompt1"],
            "prompt2": sample2["prompt2"],
            "response1": sample1["response1"],
            "response2": sample2["response2"]
        }
        
        multi_task_samples.append(multi_sample)
    
    # Combine multi-task samples with remaining single-task samples
    mixed_samples = multi_task_samples + to_keep
    random.shuffle(mixed_samples)
    
    print(f"Created dynamic mixed dataset:")
    print(f"  - Original samples: {total_samples}")
    print(f"  - Single-task samples: {len(to_keep)}")
    print(f"  - Multi-task samples: {len(multi_task_samples)}")
    print(f"  - Total samples: {len(mixed_samples)}")
    
    return mixed_samples

# ----- Chunked Dataset Processing Classes -----

class ChunkedDualModeDataset(Dataset):
    """
    Dataset that processes data in chunks to avoid memory issues.
    Handles both single-prompt and dual-prompt formats.
    """
    def __init__(self, file_path, tokenizer1, tokenizer2, max_length=512, num_workers=None, 
                 chunk_size=100000, process_chunk_size=1000, use_threads=True, 
                 cache_dir="./dataset_cache", use_cache=True):
        self.file_path = file_path
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.max_length = max_length
        self.num_workers = num_workers if num_workers is not None else max(1, multiprocessing.cpu_count() - 1)
        self.process_chunk_size = process_chunk_size
        self.use_threads = use_threads
        self.chunk_size = chunk_size  # Number of examples to process before saving
        self.cache_dir = cache_dir
        
        # Store paths rather than tokenizer objects for threading
        self.tokenizer1_path = tokenizer1.name_or_path
        self.tokenizer2_path = tokenizer2.name_or_path
        
        # Create cache base path
        cache_key = self._create_cache_key(file_path, tokenizer1, tokenizer2, max_length)
        self.cache_base_path = os.path.join(cache_dir, f"chunked_dataset_{cache_key}")
        self.metadata_path = f"{self.cache_base_path}_metadata.pkl"
        
        # Check if we have already processed metadata
        self.metadata = None
        if use_cache and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                print(f"Found existing processed data with {self.metadata['total_samples']} samples in {self.metadata['num_chunks']} chunks")
                
                # Check if all chunks exist
                all_chunks_exist = True
                for chunk_path in self.metadata['chunk_paths']:
                    if not os.path.exists(chunk_path):
                        all_chunks_exist = False
                        print(f"Missing chunk file: {chunk_path}")
                        break
                
                if not all_chunks_exist:
                    print("Some chunk files are missing, will reprocess the data")
                    self.metadata = None
            except Exception as e:
                print(f"Error loading metadata: {e}")
                self.metadata = None
        
        # If we don't have valid metadata, process the data
        if self.metadata is None:
            self._process_file_in_chunks()
        
        # Calculate total length from metadata
        self.total_length = self.metadata['total_samples']
        
        # Set up chunk index mapping for __getitem__
        self._build_index_mapping()
    
    def _build_index_mapping(self):
        """Build mapping from sample index to chunk index and internal index"""
        self.chunk_offsets = [0]
        current_offset = 0
        
        # Get sizes of each chunk
        for chunk_idx, chunk_path in enumerate(self.metadata['chunk_paths']):
            chunk_size = self.metadata['chunk_sizes'][chunk_idx]
            current_offset += chunk_size
            self.chunk_offsets.append(current_offset)
    
    def _create_cache_key(self, file_path, tokenizer1, tokenizer2, max_length):
        """Create a unique hash key for the dataset configuration"""
        # Get file modification time for the dataset file
        mtime = os.path.getmtime(file_path)
        
        # Create a string with all relevant parameters
        config_str = f"{file_path}_{mtime}_{tokenizer1.name_or_path}_{tokenizer2.name_or_path}_{max_length}"
        
        # Create a hash
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _process_file_in_chunks(self):
        """Process the input file in chunks to manage memory usage"""
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Count total lines for progress tracking
        with open(self.file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"Processing {total_lines} lines in chunks of {self.chunk_size}")
        
        # Create metadata structure
        self.metadata = {
            'total_samples': 0,
            'num_chunks': 0,
            'chunk_paths': [],
            'chunk_sizes': []
        }
        
        # Process file in chunks
        with open(self.file_path, 'r', encoding='utf-8') as f:
            chunk_idx = 0
            lines_processed = 0
            
            while lines_processed < total_lines:
                # Get the next chunk of lines
                chunk_lines = []
                chunk_line_count = min(self.chunk_size, total_lines - lines_processed)
                
                for _ in range(chunk_line_count):
                    line = f.readline()
                    if not line:  # End of file
                        break
                    chunk_lines.append(line)
                
                # If we have lines to process
                if chunk_lines:
                    # Process this chunk
                    samples = self._process_chunk(chunk_lines)
                    
                    # Create chunk path
                    chunk_path = f"{self.cache_base_path}_chunk{chunk_idx}.pkl"
                    
                    # Save the processed chunk
                    with open(chunk_path, 'wb') as cf:
                        pickle.dump(samples, cf, protocol=4)
                    
                    # Update metadata
                    self.metadata['chunk_paths'].append(chunk_path)
                    self.metadata['chunk_sizes'].append(len(samples))
                    self.metadata['total_samples'] += len(samples)
                    self.metadata['num_chunks'] += 1
                    
                    print(f"Processed and saved chunk {chunk_idx} with {len(samples)} samples")
                    
                    # Update counters
                    lines_processed += len(chunk_lines)
                    chunk_idx += 1
                
                    # Save metadata after each chunk in case of interruption
                    with open(self.metadata_path, 'wb') as mf:
                        pickle.dump(self.metadata, mf)
                else:
                    break  # No more lines to process
            
        print(f"Completed processing {lines_processed} lines into {self.metadata['num_chunks']} chunks")
        print(f"Total samples: {self.metadata['total_samples']}")
        
        # Final metadata save
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def _process_chunk(self, chunk_lines):
        """Process a chunk of lines with parallel workers"""
        # Choose executor based on preference
        executor_class = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        # Split into smaller processing batches for parallel execution
        batch_size = self.process_chunk_size
        batches = []
        for i in range(0, len(chunk_lines), batch_size):
            batches.append(chunk_lines[i:i+batch_size])
        
        # Process batches in parallel
        samples = []
        with executor_class(max_workers=self.num_workers) as executor:
            # Submit all batch processing tasks
            futures = [
                executor.submit(process_batch, batch, self.tokenizer1_path, self.tokenizer2_path, self.max_length)
                for batch in batches
            ]
            
            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures), 
                total=len(futures),
                desc="Processing batches"
            ):
                try:
                    batch_results = future.result()
                    samples.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Count modes in this chunk
        single_count = sum(1 for sample in samples if sample["mode"] == "single")
        multi_count = sum(1 for sample in samples if sample["mode"] == "multi")
        
        print(f"Chunk processed: {len(samples)} samples - {single_count} single-task, {multi_count} multi-task")
        
        return samples
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Find which chunk this index belongs to
        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_idx]
        
        # Load the chunk if it's not in memory
        chunk_path = self.metadata['chunk_paths'][chunk_idx]
        
        # Load chunk
        with open(chunk_path, 'rb') as f:
            chunk = pickle.load(f)
        
        # Get the sample from the chunk
        return chunk[local_idx]

def create_multitask_dataset_in_chunks(base_dataset_path, output_base_path, 
                                      multi_task_ratio=0.3, seed=42, chunk_size=100000):
    """
    Create a multi-task dataset from a base dataset, processing in chunks
    to manage memory usage.
    
    Args:
        base_dataset_path: Base path pattern for the chunked dataset (without metadata suffix)
        output_base_path: Base path pattern for the output multi-task dataset
        multi_task_ratio: Ratio of samples to convert to multi-task format (0-1)
        seed: Random seed for reproducibility
        chunk_size: Size of in-memory chunks to process at once
    """
    # Load the source metadata
    metadata_path = f"{base_dataset_path}_metadata.pkl"
    print(f"Loading source dataset metadata from: {metadata_path}")
    
    with open(metadata_path, 'rb') as f:
        source_metadata = pickle.load(f)
        
    total_samples = source_metadata['total_samples']
    num_chunks = source_metadata['num_chunks']
    chunk_paths = source_metadata['chunk_paths']
    chunk_sizes = source_metadata['chunk_sizes']
    
    print(f"Source dataset has {total_samples} samples in {num_chunks} chunks")
    
    # Pre-calculate random mapping for the entire dataset
    # This ensures consistent behavior regardless of how we process chunks
    import random
    random.seed(seed)
    
    # Determine how many samples we'll convert (should be even)
    num_samples_to_convert = int(total_samples * multi_task_ratio)
    if num_samples_to_convert % 2 != 0:
        num_samples_to_convert -= 1
    
    # Create a mapping of which indices to convert
    # We make an array of all indices and shuffle it
    all_indices = list(range(total_samples))
    random.shuffle(all_indices)
    
    # The first portion will be converted to multi-task
    indices_to_convert = set(all_indices[:num_samples_to_convert])
    print(f"Will convert {num_samples_to_convert} samples to multi-task format")
    
    # Initialize metadata for the multi-task dataset
    multi_metadata = {
        'total_samples': 0,
        'num_chunks': 0,
        'chunk_paths': [],
        'chunk_sizes': []
    }
    
    # Calculate how many chunks of multi-task data we'll create
    # Each output chunk will have chunk_size samples
    output_chunk_idx = 0
    samples_created = 0
    pair_buffer = []  # Buffer for samples waiting to be paired
    
    # Mapping from original index to new pair index
    pair_mapping = {}
    for i in range(0, num_samples_to_convert, 2):
        if i+1 < num_samples_to_convert:
            pair_mapping[all_indices[i]] = all_indices[i+1]
            pair_mapping[all_indices[i+1]] = all_indices[i]
    
    # Process all source chunks
    for source_chunk_idx, source_chunk_path in enumerate(chunk_paths):
        print(f"Processing source chunk {source_chunk_idx+1}/{num_chunks}...")
        
        # Load the source chunk
        with open(source_chunk_path, 'rb') as f:
            source_chunk = pickle.load(f)
        
        # Buffer for the current output chunk
        output_buffer = []
        
        # Determine the global index of the first sample in this chunk
        chunk_start_idx = sum(chunk_sizes[:source_chunk_idx])
        
        # Process each sample in the source chunk
        for local_idx, sample in enumerate(source_chunk):
            global_idx = chunk_start_idx + local_idx
            
            if global_idx in indices_to_convert:
                # This sample should be converted as part of a pair
                if global_idx in pair_mapping:
                    # Store relevant info about this sample for later pairing
                    pair_buffer.append({
                        'global_idx': global_idx,
                        'pair_idx': pair_mapping[global_idx],
                        'sample': sample
                    })
                    
                    # Check if we have both samples of a pair
                    paired_indices = [item['global_idx'] for item in pair_buffer]
                    
                    if len(pair_buffer) >= 2 and pair_mapping[global_idx] in paired_indices:
                        # Find the two samples that form a pair
                        idx1 = paired_indices.index(global_idx)
                        idx2 = paired_indices.index(pair_mapping[global_idx])
                        
                        sample1 = pair_buffer[idx1]['sample']
                        sample2 = pair_buffer[idx2]['sample']
                        
                        # Create a multi-task sample
                        multi_sample = {
                            "input_ids1": sample1["input_ids1"],
                            "attention_mask1": sample1["attention_mask1"],
                            "labels1": sample1["labels1"],
                            "input_ids2": sample2["input_ids2"],
                            "attention_mask2": sample2["attention_mask2"],
                            "labels2": sample2["labels2"],
                            "mode": "multi",
                            "prompt1": sample1["prompt1"],
                            "prompt2": sample2["prompt2"],
                            "response1": sample1["response1"],
                            "response2": sample2["response2"]
                        }
                        
                        # Add to output buffer
                        output_buffer.append(multi_sample)
                        
                        # Remove these samples from the pair buffer
                        pair_buffer = [item for i, item in enumerate(pair_buffer) 
                                  if i != idx1 and i != idx2]
                    
            else:
                # This sample remains as a single-task sample
                output_buffer.append(sample)
            
            # If the output buffer is large enough, save it
            if len(output_buffer) >= chunk_size:
                output_chunk_path = f"{output_base_path}_chunk{output_chunk_idx}.pkl"
                
                # Save the output chunk
                with open(output_chunk_path, 'wb') as f:
                    pickle.dump(output_buffer, f, protocol=4)
                
                # Update metadata
                multi_metadata['chunk_paths'].append(output_chunk_path)
                multi_metadata['chunk_sizes'].append(len(output_buffer))
                multi_metadata['total_samples'] += len(output_buffer)
                multi_metadata['num_chunks'] += 1
                
                print(f"Saved output chunk {output_chunk_idx} with {len(output_buffer)} samples")
                samples_created += len(output_buffer)
                
                # Reset the output buffer
                output_buffer = []
                output_chunk_idx += 1
    
    # Save any remaining samples
    if output_buffer:
        output_chunk_path = f"{output_base_path}_chunk{output_chunk_idx}.pkl"
        
        # Save the output chunk
        with open(output_chunk_path, 'wb') as f:
            pickle.dump(output_buffer, f, protocol=4)
        
        # Update metadata
        multi_metadata['chunk_paths'].append(output_chunk_path)
        multi_metadata['chunk_sizes'].append(len(output_buffer))
        multi_metadata['total_samples'] += len(output_buffer)
        multi_metadata['num_chunks'] += 1
        
        print(f"Saved final output chunk {output_chunk_idx} with {len(output_buffer)} samples")
        samples_created += len(output_buffer)
    
    # If there are unpaired samples in the buffer, add them as single-task samples
    if pair_buffer:
        print(f"Warning: {len(pair_buffer)} samples couldn't be paired, adding as single-task")
        
        # Get the last chunk or create a new one
        if output_buffer:
            output_buffer = []
            output_chunk_idx += 1
        
        # Add the unpaired samples as single-task
        for item in pair_buffer:
            output_buffer.append(item['sample'])
        
        output_chunk_path = f"{output_base_path}_chunk{output_chunk_idx}.pkl"
        
        # Save the output chunk
        with open(output_chunk_path, 'wb') as f:
            pickle.dump(output_buffer, f, protocol=4)
        
        # Update metadata
        multi_metadata['chunk_paths'].append(output_chunk_path)
        multi_metadata['chunk_sizes'].append(len(output_buffer))
        multi_metadata['total_samples'] += len(output_buffer)
        multi_metadata['num_chunks'] += 1
        
        print(f"Saved unpaired samples in chunk {output_chunk_idx} with {len(output_buffer)} samples")
        samples_created += len(output_buffer)
    
    # Save the metadata
    metadata_path = f"{output_base_path}_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(multi_metadata, f)
    
    print(f"Multi-task dataset creation complete!")
    print(f"Created {samples_created} samples in {multi_metadata['num_chunks']} chunks")
    print(f"Single-task: {total_samples - num_samples_to_convert}")
    print(f"Multi-task: {num_samples_to_convert // 2}")
    
    return multi_metadata

class ChunkedDatasetReader(Dataset):
    """
    A dataset reader that reads from pre-processed chunks without loading
    everything into memory at once.
    """
    def __init__(self, metadata_path):
        """
        Initialize with a path to the metadata file.
        """
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.total_samples = self.metadata['total_samples']
        self.chunk_paths = self.metadata['chunk_paths']
        self.chunk_sizes = self.metadata['chunk_sizes']
        
        # Build a mapping from sample index to chunk index and local index
        self.chunk_offsets = [0]
        for size in self.chunk_sizes:
            self.chunk_offsets.append(self.chunk_offsets[-1] + size)
        
        # Cache for loaded chunks
        self.chunk_cache = {}
        self.max_cached_chunks = 3  # Maximum number of chunks to keep in memory
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which chunk this index belongs to
        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        local_idx = idx - self.chunk_offsets[chunk_idx]
        
        # Check if the chunk is already in the cache
        if chunk_idx not in self.chunk_cache:
            # If cache is full, remove the oldest accessed chunk
            if len(self.chunk_cache) >= self.max_cached_chunks:
                oldest_chunk = min(self.chunk_cache.items(), key=lambda x: x[1][1])[0]
                del self.chunk_cache[oldest_chunk]
            
            # Load the chunk
            with open(self.chunk_paths[chunk_idx], 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Store chunk in cache with access time
            self.chunk_cache[chunk_idx] = (chunk_data, time.time())
        else:
            # Update access time for LRU
            chunk_data, _ = self.chunk_cache[chunk_idx]
            self.chunk_cache[chunk_idx] = (chunk_data, time.time())
        
        # Get the sample from the cached chunk
        return self.chunk_cache[chunk_idx][0][local_idx]

# ----- Data Configuration and Loading Helper Functions -----

def prepare_dataset_for_training_chunked(config):
    """
    Prepare dataset for training with chunked processing to manage memory usage.
    
    Args:
        config: Dictionary with configuration settings
    
    Returns:
        train_dataloader, val_dataloader, tokenizer1, tokenizer2
    """
    # Create directories if needed
    os.makedirs(config.get("dataset_cache_dir", "./dataset_cache"), exist_ok=True)
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Load tokenizers
    tokenizer1 = AutoTokenizer.from_pretrained(config["model_dir1"])
    tokenizer2 = AutoTokenizer.from_pretrained(config["model_dir2"])
    
    # Handle special tokens
    if tokenizer1.pad_token is None:
        tokenizer1.pad_token = tokenizer1.eos_token
    
    if tokenizer2.pad_token is None:
        tokenizer2.pad_token = tokenizer2.eos_token
    
    # Set chunk sizes
    processing_chunk_size = config.get("processing_chunk_size", 100000)  # Number of lines to process at once
    batch_size = config.get("data_batch_size", 1000)  # Size of batches for parallel processing
    
    # Check if we should create multi-task samples
    create_multitask = config.get("create_multitask_samples", False)
    multitask_ratio = config.get("multitask_ratio", 0.3)
    
    # Set up paths
    cache_dir = config.get("dataset_cache_dir", "./dataset_cache")
    file_path = config["dataset_path"]
    
    # Generate cache keys
    raw_cache_key = hashlib.md5(f"{file_path}_{os.path.getmtime(file_path)}_{tokenizer1.name_or_path}_{tokenizer2.name_or_path}_{config['max_length']}".encode()).hexdigest()
    
    # The ChunkedDualModeDataset uses "chunked_dataset_" as prefix
    chunked_base_path = os.path.join(cache_dir, f"chunked_dataset_{raw_cache_key}")
    chunked_metadata_path = f"{chunked_base_path}_metadata.pkl"
    
    # Multi-task key
    multi_cache_key = hashlib.md5(f"{raw_cache_key}_{multitask_ratio}_{config.get('seed', 42)}".encode()).hexdigest()
    multi_base_path = os.path.join(cache_dir, f"multitask_dataset_{multi_cache_key}")
    multi_metadata_path = f"{multi_base_path}_metadata.pkl"
    
    print(f"Loading dataset from {file_path}...")
    start_time = time.time()
    
    # Step 1: Process the raw dataset if needed
    if not os.path.exists(chunked_metadata_path) or not config.get("use_cached_dataset", True):
        print("Raw dataset not found in cache, processing from scratch...")
        
        # Process the dataset in chunks
        raw_dataset = ChunkedDualModeDataset(
            file_path,
            tokenizer1,
            tokenizer2,
            max_length=config["max_length"],
            num_workers=config.get("data_workers", None),
            chunk_size=processing_chunk_size,
            process_chunk_size=batch_size,
            use_threads=config.get("use_threads", True),
            cache_dir=cache_dir,
            use_cache=config.get("use_cached_dataset", True)
        )
        
        # Update path to match the actual generated path
        chunked_metadata_path = raw_dataset.metadata_path
        chunked_base_path = chunked_metadata_path.replace("_metadata.pkl", "")
        
        print(f"Raw dataset processing complete, metadata saved to {chunked_metadata_path}")
    else:
        print(f"Using cached raw dataset: {chunked_metadata_path}")
    
    # Step 2: Create multi-task dataset if needed
    final_metadata_path = None
    if create_multitask:
        if not os.path.exists(multi_metadata_path) or not config.get("use_cached_dataset", True):
            print(f"Creating multi-task dataset with ratio {multitask_ratio}...")
            
            # Create multi-task dataset from the raw dataset
            # Pass the correct base path that matches the raw dataset's actual path
            multi_metadata = create_multitask_dataset_in_chunks(
                chunked_base_path,  # Use the chunked base path, not raw_base_path
                multi_base_path,
                multi_task_ratio=multitask_ratio,
                seed=config.get("seed", 42),
                chunk_size=processing_chunk_size
            )
            
            final_metadata_path = multi_metadata_path
        else:
            print(f"Using cached multi-task dataset: {multi_metadata_path}")
            final_metadata_path = multi_metadata_path
    else:
        # Use the raw dataset as the final dataset
        final_metadata_path = chunked_metadata_path
    
    # Step 3: Create a dataset reader for the final dataset
    dataset = ChunkedDatasetReader(final_metadata_path)
    
    total_load_time = time.time() - start_time
    print(f"Dataset preparation completed in {total_load_time:.2f} seconds")
    print(f"Total samples: {len(dataset)}")
    
    # Split dataset into train and validation
    dataset_size = len(dataset)
    train_size = int(config["train_val_split"] * dataset_size)
    val_size = dataset_size - train_size
    
    # Use a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(config.get("seed", 42))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config["train_batch_size"],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config["eval_batch_size"]
    )
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    return train_dataloader, val_dataloader, tokenizer1, tokenizer2

# ----- Legacy Functions (Kept for Backward Compatibility) -----

def load_dataset_sequential(file_path, tokenizer1, tokenizer2, max_length=512):
    """
    Load dataset sequentially in a single process.
    Use this as a fallback if parallel processing fails.
    """
    samples = []
    
    # Read all lines from file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_samples = len(lines)
    print(f"Loading dataset sequentially with {total_samples} samples...")
    
    # Process one line at a time
    for line in tqdm(lines, desc="Tokenizing dataset"):
        try:
            format_type = detect_sample_format(line)
            
            if format_type == "single":
                sample = process_single_task_sample(line, tokenizer1, tokenizer2, max_length)
            else:
                sample = process_multi_task_sample(line, tokenizer1, tokenizer2, max_length)
                
            samples.append(sample)
        except Exception as e:
            print(f"Error processing line: {e}")
    
    # Count modes
    single_count = sum(1 for sample in samples if sample["mode"] == "single")
    multi_count = sum(1 for sample in samples if sample["mode"] == "multi")
    
    print(f"Dataset loaded successfully: {len(samples)} samples")
    print(f"  - Single-task samples: {single_count}")
    print(f"  - Multi-task samples: {multi_count}")
    
    return samples

def create_mixed_dataset(single_file, multi_file, output_file, ratio=0.5):
    """
    Create a mixed dataset with both single and multi-task samples.
    
    Args:
        single_file: Path to the single-task format file
        multi_file: Path to the multi-task format file
        output_file: Path to save the mixed dataset
        ratio: Ratio of single-task samples (0-1)
    """
    # Read files
    with open(single_file, 'r', encoding='utf-8') as f:
        single_lines = f.readlines()
    
    with open(multi_file, 'r', encoding='utf-8') as f:
        multi_lines = f.readlines()
    
    # Calculate counts
    total_count = len(single_lines) + len(multi_lines)
    single_count = int(total_count * ratio)
    multi_count = total_count - single_count
    
    # Adjust if needed
    if single_count > len(single_lines):
        single_count = len(single_lines)
        multi_count = total_count - single_count
    
    if multi_count > len(multi_lines):
        multi_count = len(multi_lines)
        single_count = total_count - multi_count
    
    # Select samples
    selected_single = single_lines[:single_count]
    selected_multi = multi_lines[:multi_count]
    
    # Shuffle
    import random
    combined = selected_single + selected_multi
    random.shuffle(combined)
    
    # Write output
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in combined:
            f.write(line)
    
    print(f"Created mixed dataset with {single_count} single-task and {multi_count} multi-task samples")
    print(f"Output saved to {output_file}")

def convert_single_to_multi_format(input_file, output_file, mode="duplicate"):
    """
    Convert a single-task format file to multi-task format.
    
    Modes:
    - duplicate: Use the same prompt/response for both tasks
    - alternate: Alternate prompts and responses (1 with 2, 2 with 3, etc.)
    - shuffle: Randomly match prompts with different responses
    """
    import random
    
    # Read all samples
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    samples = [json.loads(line) for line in lines]
    output_samples = []
    
    if mode == "duplicate":
        # Use the same prompt/response for both tasks
        for sample in samples:
            output_samples.append({
                "prompt1": sample["prompt"],
                "response1": sample["response"],
                "prompt2": sample["prompt"],
                "response2": sample["response"]
            })
    
    elif mode == "alternate":
        # Alternate prompts and responses
        for i in range(len(samples) - 1):
            output_samples.append({
                "prompt1": samples[i]["prompt"],
                "response1": samples[i]["response"],
                "prompt2": samples[i+1]["prompt"],
                "response2": samples[i+1]["response"]
            })
    
    elif mode == "shuffle":
        # Randomly match prompts with different responses
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        for i in range(len(samples)):
            output_samples.append({
                "prompt1": samples[i]["prompt"],
                "response1": samples[i]["response"],
                "prompt2": shuffled_samples[i]["prompt"],
                "response2": shuffled_samples[i]["response"]
            })
    
    # Write the output
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in output_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Converted {len(samples)} samples to multi-task format using {mode} mode")
    print(f"Output saved to {output_file}")
