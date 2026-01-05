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
        - audio: numpy array of audio samples
        - label/targets: numpy array of frame-level labels
        - (other fields like file_path, sample_rate, metadata are optional)
    """
    
    def __init__(
        self,
        parquet_dir: str,
        feature_extractor=None,
        max_length_seconds: float = 10.0,
        sample_rate: int = 16000,
        shuffle: bool = True,
        seed: int = 42,
        streaming: bool = False,
        remove_deletions: bool = False,
        remove_substitutions: bool = False,
        remove_insertions: bool = False,
        cache_dir: Optional[str] = None,
        label_column: str = "targets",  # Can be "targets" or "label"
        return_file_path: bool = False,  # Return file_path for decoding
        require_labels: bool = True,  # If False, labels are optional (for inference)
        word_level: bool = False,
    ):
        """
        Args:
            parquet_dir: Directory containing parquet files
            feature_extractor: HuBERT feature extractor (for preprocessing)
            max_length_seconds: Maximum audio length in seconds
            sample_rate: Audio sample rate
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            streaming: If True, uses streaming mode for very large datasets (saves memory)
            cache_dir: Directory to cache the dataset (default: ~/.cache/huggingface/datasets)
            label_column: Name of the label column in parquet files ("targets" or "label")
        """
        self.parquet_dir = Path(parquet_dir)
        self.feature_extractor = feature_extractor
        self.max_length_seconds = max_length_seconds
        self.sample_rate = sample_rate
        self.max_length_samples = int(max_length_seconds * sample_rate)
        self.shuffle = shuffle
        self.seed = seed
        self.streaming = streaming
        self.remove_deletions = remove_deletions
        self.remove_substitutions = remove_substitutions
        self.remove_insertions = remove_insertions
        self.label_column = label_column
        self.return_file_path = return_file_path
        self.require_labels = require_labels
        self.word_level = word_level
                
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
                - input_values: preprocessed audio tensor
                - labels: frame-level label tensor
                - attention_mask: attention mask tensor
        """
        # Get sample from HuggingFace dataset
        # HF datasets handle caching and memory mapping automatically
        sample = self.dataset[idx]
        
        # Extract audio
        audio = sample['audio']
        uttid = sample['uttid']
        # Handle labels (optional for inference)
        labels = sample.get('targets', sample.get('label', None))
        labels = labels.get('labels', None)
        # labels are a list of dicts [{"frame_idx": 3, "label": 0}, ...]
        if labels is None and self.require_labels:
            print(f"Warning: No labels found for sample {uttid}")
            return None
        
        # # Convert to numpy arrays if needed
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Store original audio length for label truncation
        original_audio_length = len(audio)
        audio_length_in_seconds = original_audio_length / self.sample_rate
        
        # Skip samples with audio longer than max_length
        if len(audio) > self.max_length_samples:
            print(f"Warning: Skipping sample {uttid} - audio length {audio_length_in_seconds:.2f}s exceeds max {self.max_length_seconds:.2f}s")
            return None
        
        estimated_label_length = math.ceil(audio_length_in_seconds / 0.08) + 1
        
        # Preprocess audio with feature extractor
        if self.feature_extractor is not None:
            inputs = self.feature_extractor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=False
            )
            input_values = inputs.input_values.squeeze(0)
        else:
            input_values = torch.from_numpy(audio)
        
        # Create attention mask (all ones since we have real audio)
        attention_mask = torch.ones_like(input_values)
        
        result = {
            'uttid': uttid,
            'input_values': input_values,
            'attention_mask': attention_mask
        }
        
        # Add labels if available
        if labels is not None:
            # result['labels'] = labels
            if not self.word_level:
                new_labels = torch.full((estimated_label_length,), -100, dtype=torch.long)
            else:
                new_labels = torch.full((estimated_label_length,), 0, dtype=torch.long)
            
            for label in labels:
                if self.remove_deletions and label['label'] == 2:
                    continue
                if self.remove_substitutions and self.remove_insertions:
                    if self.word_level:
                        if label['label'] == 2:
                            label['label'] = 1
                    else:
                        if label['label'] == 1:
                            continue
                        elif label['label'] == 2:
                            label['label'] = 1
                new_labels[label['frame_idx']] = label['label']
            result['labels'] = new_labels
        
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
        feature_extractor=None,
        max_length_seconds: float = 10.0,
        sample_rate: int = 16000,
        shuffle: bool = True,
        seed: int = 42,
        buffer_size: int = 10000,
        cache_dir: Optional[str] = None,
        label_column: str = "targets",
    ):
        """
        Args:
            parquet_dir: Directory containing parquet files
            feature_extractor: HuBERT feature extractor (for preprocessing)
            max_length_seconds: Maximum audio length in seconds
            sample_rate: Audio sample rate
            shuffle: Whether to shuffle the dataset
            seed: Random seed for shuffling
            buffer_size: Buffer size for shuffling (larger = better shuffle, more memory)
            cache_dir: Directory to cache the dataset
            label_column: Name of the label column in parquet files
        """
        super().__init__()
        
        self.parquet_dir = Path(parquet_dir)
        self.feature_extractor = feature_extractor
        self.max_length_seconds = max_length_seconds
        self.sample_rate = sample_rate
        self.max_length_samples = int(max_length_seconds * sample_rate)
        self.shuffle = shuffle
        self.seed = seed
        self.buffer_size = buffer_size
        self.label_column = label_column
        
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
            # Extract audio and labels
            audio = sample['audio']
            
            # Handle different label column names
            if self.label_column in sample:
                labels = sample[self.label_column]
            elif 'label' in sample:
                labels = sample['label']
            elif 'targets' in sample:
                labels = sample['targets']
            else:
                continue  # Skip if no valid label column
            
            # Convert to numpy arrays if needed
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            
            # Ensure audio is float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Handle stereo audio
            if len(audio.shape) > 1:
                audio = audio.mean(axis=-1)
            
            # Store original audio length for label truncation
            original_audio_length = len(audio)
            
            # Skip samples with audio longer than max_length
            if len(audio) > self.max_length_samples:
                continue  # Skip this sample
            
            # Preprocess audio with feature extractor
            if self.feature_extractor is not None:
                inputs = self.feature_extractor(
                    audio,
                    sampling_rate=self.sample_rate,
                    return_tensors="pt",
                    padding=False
                )
                input_values = inputs.input_values.squeeze(0)
            else:
                input_values = torch.from_numpy(audio)
            
            # Convert labels to tensor
            labels_tensor = torch.from_numpy(labels).long()
            
            # Create attention mask (all ones since we have real audio)
            attention_mask = torch.ones_like(input_values)
            
            yield {
                'input_values': input_values,
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
    
    # Find max lengths in batch
    max_audio_length = max(item['input_values'].size(0) for item in batch)
    
    # Check if batch has labels
    has_labels = 'labels' in batch[0]
    if has_labels:
        max_label_length = max(item['labels'].size(0) for item in batch)
    
    # Check if batch has file_path
    has_file_path = 'file_path' in batch[0]
    
    batch_size = len(batch)
    
    # Initialize padded tensors
    input_values = torch.zeros(batch_size, max_audio_length)
    attention_mask = torch.zeros(batch_size, max_audio_length)
    
    if has_labels:
        labels = torch.full((batch_size, max_label_length), -100, dtype=torch.long)
    
    file_paths = []
    audio_lengths = []
    label_lengths = []
    utt_ids = []
    
    # Fill in the data and store utt_id
    for i, item in enumerate(batch):
        audio_len = item['input_values'].size(0)
        utt_id = item['uttid']
        input_values[i, :audio_len] = item['input_values']
        attention_mask[i, :audio_len] = item['attention_mask']
        audio_lengths.append(audio_len)
        utt_ids.append(utt_id)
        if has_labels:
            label_len = item['labels'].size(0)
            labels[i, :label_len] = item['labels']
            label_lengths.append(label_len)
        
        if has_file_path:
            file_paths.append(item['file_path'])
    
    result = {
        'input_values': input_values,
        'attention_mask': attention_mask,
        'audio_lengths': audio_lengths,
        'uttid': utt_ids,
    }
    
    if has_labels:
        result['labels'] = labels
        result['label_lengths'] = label_lengths
    
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
        sample_rate=16000,
        shuffle=True,
        seed=42,
        streaming=False,  # Use memory-mapped access
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio shape: {sample['input_values'].shape}")
    print(f"Labels shape: {sample['labels'].shape}")
    
    # Example 2: Iterable dataset for very large datasets
    print("\n=== Example 2: Iterable Dataset ===")
    iterable_dataset = HFParquetDistortionIterableDataset(
        parquet_dir="data/parquet",
        max_length_seconds=10.0,
        sample_rate=16000,
        shuffle=True,
        seed=42,
        buffer_size=10000,
    )
    
    # Test iterating
    for i, sample in enumerate(iterable_dataset):
        print(f"Sample {i} keys: {sample.keys()}")
        print(f"Audio shape: {sample['input_values'].shape}")
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
    print(f"Batch audio shape: {batch['input_values'].shape}")
    print(f"Batch labels shape: {batch['labels'].shape}")
