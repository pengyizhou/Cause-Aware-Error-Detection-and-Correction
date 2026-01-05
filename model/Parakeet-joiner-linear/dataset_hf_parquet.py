"""
PyTorch Dataset for loading distortion detection data from parquet files using HuggingFace datasets.

This dataset uses HuggingFace's datasets library for efficient parquet handling.
Provides better memory efficiency, built-in shuffling, and streaming capabilities.
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Dict, Any
from datasets import load_dataset, Dataset as HFDataset
from datasets import concatenate_datasets
import ipdb


class HFParquetDistortionDataset(Dataset):
    """
    Dataset for loading distortion detection data from parquet files using HuggingFace datasets.
    
    Benefits over manual parquet loading:
    - Built-in efficient shuffling with memory mapping
    - Automatic memory management and caching
    - Streaming support for very large datasets
    - Better multi-worker support in DataLoader
    
    Expected parquet format:
        - uttid: utterance ID (string)
        - embedding: numpy array [time, dim] - embedding matrix
        - targets: dict with:
            - raw: raw labels string
            - labels: list of dicts with frame_idx and label
            - label_array: numpy array [time] - frame-level labels
    """
    
    def __init__(
        self,
        parquet_dir: str,
        max_length_seconds: float = 10.0,
        frame_duration_seconds: float = 0.08,  # 80ms per frame (typical for speech embeddings)
        shuffle: bool = True,
        seed: int = 42,
        streaming: bool = False,
        cache_dir: Optional[str] = None,
        return_file_path: bool = False,  # Return file_path for decoding
        require_labels: bool = True,  # If False, labels are optional (for inference)
    ):
        """
        Args:
            parquet_dir: Directory containing parquet files
            max_length_seconds: Maximum embedding sequence length in seconds (converted to frames)
            frame_duration_seconds: Duration of each embedding frame in seconds (default: 0.08 = 80ms)
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            streaming: If True, uses streaming mode for very large datasets (saves memory)
            cache_dir: Directory to cache the dataset (default: ~/.cache/huggingface/datasets)
            return_file_path: Whether to return file_path in samples (for decoding)
            require_labels: If False, labels are optional (for inference)
        """
        self.parquet_dir = Path(parquet_dir)
        self.max_length_seconds = max_length_seconds
        self.frame_duration_seconds = frame_duration_seconds
        # Convert max_length_seconds to max number of frames
        self.max_length_samples = int(max_length_seconds / frame_duration_seconds)
        self.shuffle = shuffle
        self.seed = seed
        self.streaming = streaming
        self.return_file_path = return_file_path
        self.require_labels = require_labels
        
        # Find all parquet files
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {parquet_dir}")
        
        print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")
        
        # Load dataset using HuggingFace datasets
        # This automatically handles memory mapping and efficient loading
        data_files = [str(f) for f in parquet_files]
        
        print(f"Loading dataset from parquet files...")
        if streaming:
            # Streaming mode: data is loaded on-the-fly, minimal memory footprint
            print("Using streaming mode for memory efficiency")
            self.dataset = load_dataset(
                'parquet',
                data_files=data_files,
                split='train',
                streaming=True,
                cache_dir=cache_dir,
            )
            
            if shuffle:
                # Shuffle with buffer for streaming datasets
                buffer_size = 10000  # Adjust based on available memory
                self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
                print(f"Enabled streaming shuffle with buffer size {buffer_size}")
            
            # For streaming datasets, we can't get exact length upfront
            # Will be computed as we iterate
            self._length = None
            
        else:
            # Standard mode: dataset is memory-mapped, very efficient
            print("Using memory-mapped mode for fast random access")
            self.dataset = load_dataset(
                'parquet',
                data_files=data_files,
                split='train',
                cache_dir=cache_dir,
            )
            
            if shuffle:
                # Shuffle the entire dataset
                self.dataset = self.dataset.shuffle(seed=seed)
                print(f"Shuffled dataset with seed {seed}")
            
            self._length = len(self.dataset)
            print(f"Loaded {self._length} samples")
        
        # Set format to numpy for faster processing
        if not streaming:
            self.dataset.set_format(type=None)  # Keep original format for flexibility
    
    def __len__(self):
        """
        Returns the number of samples in the dataset.
        For streaming datasets, returns a large number as exact length may not be known.
        """
        if self._length is not None:
            return self._length
        else:
            # For streaming datasets, return a large number
            # This is a limitation of streaming mode
            return 1_000_000_000  # Placeholder for streaming
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            dict with keys:
                - embedding: embedding tensor [time, dim]
                - labels: frame-level label tensor [time]
                - attention_mask: attention mask tensor [time]
                - uttid: utterance ID
        """
        # Get sample from HuggingFace dataset
        # HF datasets handle caching and memory mapping automatically
        sample = self.dataset[idx]
        
        # Extract embedding and uttid
        embedding = sample['embedding']
        uttid = sample['uttid']
        
        # Extract targets dict
        targets = sample.get('targets', {})
        
        # Extract label_array from targets (this is the pre-computed label array)
        label_array = targets.get('label_array', None)
        
        # Handle labels (optional for inference)
        if label_array is None and self.require_labels:
            print(f"Warning: No labels found for sample {uttid}")
            return None
        
        # Convert embedding to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)
        
        # Ensure embedding is float32
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Get embedding dimensions
        embedding_time = embedding.shape[0]
        
        # Skip samples with embedding longer than max_length (if max_length is set)
        if hasattr(self, 'max_length_samples') and embedding_time > self.max_length_samples:
            embedding_time_seconds = embedding_time * self.frame_duration_seconds
            print(f"Warning: Skipping sample {uttid} - embedding time {embedding_time_seconds:.2f}s exceeds max {self.max_length_seconds:.2f}s")
            return None
        
        # Convert embedding to tensor [time, dim]
        embedding_tensor = torch.from_numpy(embedding)
        
        # Create attention mask (all ones since we have real embeddings)
        attention_mask = torch.ones(embedding_time, dtype=torch.long)
        
        result = {
            'uttid': uttid,
            'embedding': embedding_tensor,
            'attention_mask': attention_mask
        }
        
        # Add labels if available
        if label_array is not None:
            # Convert label_array to numpy array if needed
            if not isinstance(label_array, np.ndarray):
                label_array = np.array(label_array)
            
            # Ensure label_array matches embedding time dimension
            if len(label_array) != embedding_time:
                print(f"Warning: Label length {len(label_array)} != embedding time {embedding_time} for {uttid}")
                return None
            
            # Convert to tensor
            labels_tensor = torch.from_numpy(label_array).long()
            result['labels'] = labels_tensor
        
        # Add file_path if requested (for decoding)
        if self.return_file_path:
            file_path = sample.get('file_path', sample.get('path', f"sample_{idx}"))
            result['file_path'] = file_path
        
        return result


class HFParquetDistortionIterableDataset(torch.utils.data.IterableDataset):
    """
    Iterable version of HFParquetDistortionDataset for streaming large datasets.
    
    This is recommended for very large datasets that don't fit in memory.
    Works seamlessly with PyTorch DataLoader and supports multi-worker loading.
    """
    
    def __init__(
        self,
        parquet_dir: str,
        max_length_seconds: float = 10.0,
        frame_duration_seconds: float = 0.08,  # 80ms per frame (typical for speech embeddings)
        shuffle: bool = True,
        seed: int = 42,
        buffer_size: int = 10000,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            parquet_dir: Directory containing parquet files
            max_length_seconds: Maximum embedding sequence length in seconds (converted to frames)
            frame_duration_seconds: Duration of each embedding frame in seconds (default: 0.08 = 80ms)
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            buffer_size: Buffer size for shuffling (larger = better shuffle, more memory)
            cache_dir: Directory to cache the dataset
        """
        super().__init__()
        
        self.parquet_dir = Path(parquet_dir)
        self.max_length_seconds = max_length_seconds
        self.frame_duration_seconds = frame_duration_seconds
        # Convert max_length_seconds to max number of frames
        self.max_length_samples = int(max_length_seconds / frame_duration_seconds)
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        
        # Find all parquet files
        parquet_files = sorted(self.parquet_dir.glob("*.parquet"))
        
        if not parquet_files:
            raise ValueError(f"No parquet files found in {parquet_dir}")
        
        print(f"Found {len(parquet_files)} parquet files in {parquet_dir}")
        
        # Load dataset in streaming mode
        data_files = [str(f) for f in parquet_files]
        
        print(f"Loading iterable dataset from parquet files...")
        self.dataset = load_dataset(
            'parquet',
            data_files=data_files,
            split='train',
            streaming=True,
            cache_dir=cache_dir,
        )
        
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
            print(f"Enabled streaming shuffle with buffer size {buffer_size}")
    
    def __iter__(self):
        """
        Iterate over the dataset.
        Handles multi-worker DataLoader automatically.
        """
        # HuggingFace datasets handles worker splitting automatically
        for sample in self.dataset:
            # Extract embedding and uttid
            embedding = sample.get('embedding', None)
            uttid = sample.get('uttid', None)
            
            if embedding is None:
                continue  # Skip if no embedding
            
            # Extract targets dict
            targets = sample.get('targets', {})
            
            # Extract label_array from targets
            label_array = targets.get('label_array', None)
            
            if label_array is None:
                continue  # Skip if no labels
            
            # Convert to numpy arrays if needed
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            if not isinstance(label_array, np.ndarray):
                label_array = np.array(label_array)
            
            # Ensure embedding is float32
            if embedding.dtype != np.float32:
                embedding = embedding.astype(np.float32)
            
            # Get embedding dimensions
            embedding_time = embedding.shape[0]
            
            # Skip samples with embedding longer than max_length
            if embedding_time > self.max_length_samples:
                continue  # Skip this sample
            
            # Ensure label_array matches embedding time dimension
            if len(label_array) != embedding_time:
                continue  # Skip if dimensions don't match
            
            # Convert embedding to tensor [time, dim]
            embedding_tensor = torch.from_numpy(embedding)
            
            # Convert labels to tensor
            labels_tensor = torch.from_numpy(label_array).long()
            
            # Create attention mask (all ones since we have real embeddings)
            attention_mask = torch.ones(embedding_time, dtype=torch.long)
            
            yield {
                'uttid': uttid if uttid else f"sample_{id(sample)}",
                'embedding': embedding_tensor,
                'labels': labels_tensor,
                'attention_mask': attention_mask
            }


def collate_fn_hf_parquet(batch):
    """
    Collate function for HuggingFace parquet dataset.
    Pads sequences to the same length within a batch.
    
    Compatible with both map-style and iterable datasets.
    Filters out None samples (samples with missing labels).
    Supports optional labels and file_path (for decoding).
    """
    # Filter out None samples (samples with missing labels)
    batch = [item for item in batch if item is not None]
    
    # If entire batch is None, return None (training loop should skip)
    if len(batch) == 0:
        return None
    
    # Find max time dimension in batch (embeddings are [time, dim])
    max_time = max(item['embedding'].size(0) for item in batch)
    embedding_dim = batch[0]['embedding'].size(1)  # Get embedding dimension
    
    # Check if batch has labels
    has_labels = 'labels' in batch[0]
    if has_labels:
        max_label_length = max(item['labels'].size(0) for item in batch)
        # Ensure max_label_length matches max_time (they should be the same)
        if max_label_length != max_time:
            max_time = max(max_time, max_label_length)
    
    # Check if batch has file_path
    has_file_path = 'file_path' in batch[0]
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    # Embedding: [batch_size, max_time, embedding_dim]
    embeddings = torch.zeros(batch_size, max_time, embedding_dim)
    attention_mask = torch.zeros(batch_size, max_time, dtype=torch.long)
    
    if has_labels:
        labels = torch.full((batch_size, max_time), -100, dtype=torch.long)
    
    file_paths = []
    embedding_lengths = []
    label_lengths = []
    utt_ids = []
    
    # Fill in the data and store utt_id
    for i, item in enumerate(batch):
        embedding_time = item['embedding'].size(0)
        utt_id = item['uttid']
        
        # Copy embedding [time, dim] -> [max_time, dim] with padding
        embeddings[i, :embedding_time, :] = item['embedding']
        attention_mask[i, :embedding_time] = item['attention_mask']
        embedding_lengths.append(embedding_time)
        utt_ids.append(utt_id)
        
        if has_labels:
            label_len = item['labels'].size(0)
            labels[i, :label_len] = item['labels']
            label_lengths.append(label_len)
        
        if has_file_path:
            file_paths.append(item['file_path'])
    
    result = {
        'embedding': embeddings,
        'attention_mask': attention_mask,
        'embedding_lengths': embedding_lengths,
        'uttid': utt_ids,
    }
    
    if has_labels:
        result['labels'] = labels
        result['label_lengths'] = label_lengths
    
    if has_file_path:
        result['file_path'] = file_paths
    
    return result

# Example usage
if __name__ == "__main__":
    """
    Example usage of the HuggingFace parquet dataset classes.
    """
    
    # Example 1: Standard map-style dataset (recommended for most use cases)
    print("\n=== Example 1: Map-style Dataset ===")
    dataset = HFParquetDistortionDataset(
        parquet_dir="data/parquet",
        max_length_seconds=10.0,
        frame_duration_seconds=0.08,  # 80ms per frame
        shuffle=True,
        seed=42,
        streaming=False,  # Use memory-mapped access
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Embedding shape: {sample['embedding'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    print(f"Uttid: {sample['uttid']}")
    
    # Example 2: Iterable dataset for very large datasets
    print("\n=== Example 2: Iterable Dataset ===")
    iterable_dataset = HFParquetDistortionIterableDataset(
        parquet_dir="data/parquet",
        max_length_seconds=10.0,
        frame_duration_seconds=0.08,
        shuffle=True,
        seed=42,
        buffer_size=10000,
    )
    
    # Test iterating
    for i, sample in enumerate(iterable_dataset):
        print(f"Sample {i} keys: {sample.keys()}")
        print(f"Embedding shape: {sample['embedding'].shape}")
        print(f"Labels shape: {sample['labels'].shape}")
        if i >= 2:  # Just show first 3 samples
            break
    
    # Example 3: Using with DataLoader
    print("\n=== Example 3: DataLoader ===")
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,  # Shuffle already done in dataset
        num_workers=2,
        collate_fn=collate_fn_hf_parquet,
    )
    
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch embedding shape: {batch['embedding'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
