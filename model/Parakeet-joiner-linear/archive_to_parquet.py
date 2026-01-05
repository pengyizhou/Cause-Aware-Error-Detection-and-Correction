#!/usr/bin/env python3
"""
Archive embedding data from joint_embeddings.pt and labels.txt to parquet format.
Only includes utterances that exist in both files and where embedding time matches label length.

Output format: $output_path/$data_name/${data_name}_00001.parquet
Each record contains: uttid, embedding (numpy array [time, dim]), targets (dict with label_array)

If embedding time doesn't match label length, the uttid is printed and skipped.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import ipdb


def load_joint_embeddings(embeddings_path: str) -> Dict[str, np.ndarray]:
    """
    Load joint embeddings from .pt file.
    
    Args:
        embeddings_path: Path to joint_embeddings.pt file
        
    Returns:
        Dict mapping uttid to embedding matrix [time, dim]
    """
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings_dict = torch.load(embeddings_path, map_location='cpu')
    
    # Convert torch tensors to numpy arrays
    uttid_to_embedding = {}
    for uttid, embedding in embeddings_dict.items():
        if isinstance(embedding, torch.Tensor):
            uttid_to_embedding[uttid] = embedding.cpu().numpy()
        else:
            uttid_to_embedding[uttid] = np.array(embedding)
    
    print(f"  Loaded {len(uttid_to_embedding)} embeddings")
    return uttid_to_embedding


def deduplicate_labels(parsed_labels: List[Dict]) -> List[Dict]:
    """
    Remove duplicate entries from parsed_labels list.
    If same frame_idx appears multiple times, keeps the last occurrence.
    
    Args:
        parsed_labels: List of dicts with 'frame_idx' and 'label' keys
        
    Returns:
        Deduplicated list, sorted by frame_idx
    """
    # Use dict to deduplicate by frame_idx (last occurrence wins)
    unique_dict = {}
    for item in parsed_labels:
        frame_idx = item['frame_idx']
        unique_dict[frame_idx] = item
    
    # Convert back to list and sort by frame_idx to maintain order
    deduplicated = sorted(unique_dict.values(), key=lambda x: x['frame_idx'])
    return deduplicated


def parse_labels_txt(labels_path: str, label_reformed: str = "-1", add_rir: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Parse labels.txt file to get uttid -> labels dict mapping.
    Labels format: uttid\t[frame_idx:label,...]
    
    Args:
        labels_path: Path to labels.txt file
        label_reformed: Label to re-form the labels.txt file (default: -1, unchanged). 
                        0: Clean, 1: Noisy, 2: RIR, 3: Interference, 4: Packet Loss, 5: Missing, 
                        6: RIR+Noise, 7: RIR+Interference
        add_rir: Add RIR to the final labels (default: False)
        
    Returns:
        Dict mapping uttid to targets dict with 'labels' key containing the parsed list
    """
    uttid_to_targets = {}
    with open(labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by tab or single space
            if '\t' in line:
                parts = line.split('\t', 1)
            else:
                parts = line.split(' ', 1)
            if len(parts) >= 2:
                uttid = parts[0]
                labels_str = parts[1]
                
                # Parse the labels list [frame:label, frame:label, ...]
                # Store as dict with raw string and parsed list
                parsed_labels = []
                if labels_str.startswith('[') and labels_str.endswith(']'):
                    labels_content = labels_str[1:-1]  # Remove brackets
                    if labels_content:
                        for item in labels_content.split(','):
                            item = item.strip()
                            if ':' in item:
                                frame_idx, label = item.split(':')
                                if label_reformed == "-1":
                                    if not add_rir:
                                        parsed_labels.append({
                                            'frame_idx': int(frame_idx),
                                            'label': int(label)
                                        })
                                    else:
                                        if int(label) == 1:
                                            parsed_labels.append({
                                                'frame_idx': int(frame_idx),
                                                'label': 6
                                            })
                                        elif int(label) == 3:
                                            parsed_labels.append({
                                                'frame_idx': int(frame_idx),
                                                'label': 7
                                            })
                                        else:
                                            parsed_labels.append({
                                                'frame_idx': int(frame_idx),
                                                'label': int(label)
                                            })
                                else:
                                    if int(label) == 0:
                                        parsed_labels.append({
                                            'frame_idx': int(frame_idx),
                                            'label': int(label)
                                        })
                                    elif int(label) == 1:
                                        parsed_labels.append({
                                            'frame_idx': int(frame_idx),
                                            'label': int(label_reformed)
                                        })
                
                # Deduplicate labels (remove duplicates with same frame_idx)
                # parsed_labels = deduplicate_labels(parsed_labels)
                
                uttid_to_targets[uttid] = {
                    'raw': labels_str,
                    'labels': parsed_labels
                }
    return uttid_to_targets


def verify_and_extract_labels(
    embedding: np.ndarray, 
    parsed_labels: List[Dict], 
    uttid: str, 
    verbose: bool = True
) -> Tuple[np.ndarray, bool]:
    """
    Verify if embedding time dimension matches label count.
    If matches, extract labels as numpy array [0, 1, 0, 1, ...], otherwise print uttid.
    
    Args:
        embedding: Embedding matrix [time, dim]
        parsed_labels: List of dicts with 'frame_idx' and 'label' keys (frame_idx ignored, labels are in order)
        uttid: Utterance ID
        verbose: Whether to print mismatches
        
    Returns:
        Tuple of (label_array, is_valid) where is_valid indicates if time matches
    """
    # ipdb.set_trace()
    embedding_time = embedding.shape[0]
    label_count = len(parsed_labels)
    
    # Simply check if the number of labels matches embedding time dimension
    if embedding_time != label_count:
        parsed_labels = deduplicate_labels(parsed_labels)
        label_count = len(parsed_labels)
        if embedding_time != label_count:
            if verbose:
                print(f"Mismatch for {uttid}: embedding_time={embedding_time}, label_count={label_count}")
            return np.array([], dtype=np.int32), False
    
    # Time matches, extract labels directly (labels are already in order)
    if parsed_labels:
        label_array = np.array([item['label'] for item in parsed_labels], dtype=np.int32)
    else:
        label_array = np.array([], dtype=np.int32)
    
    return label_array, True


def process_and_save_batch(
    batch_uttids: List[str],
    uttid_to_embeddings: Dict[str, np.ndarray],
    uttid_to_targets: Dict[str, Dict],
    output_path: str,
    verbose: bool = True
) -> Tuple[str, float, int]:
    """
    Process a batch of utterances and save to parquet.
    
    Args:
        batch_uttids: List of uttids to process
        uttid_to_embeddings: Dict mapping uttid to embedding matrix [time, dim]
        uttid_to_targets: Dict mapping uttid to targets dict
        output_path: Full output file path
        verbose: Whether to print progress
        
    Returns:
        Tuple of (output_file_path, file_size_MB, num_records)
    """
    records = []
    mismatched_uttids = []
    
    for uttid in batch_uttids:
        if uttid not in uttid_to_embeddings:
            if verbose:
                print(f"  Warning: Embedding not found for {uttid}")
            continue
            
        if uttid not in uttid_to_targets:
            if verbose:
                print(f"  Warning: Labels not found for {uttid}")
            continue
        
        embedding = uttid_to_embeddings[uttid]
        targets = uttid_to_targets[uttid]
        parsed_labels = targets.get('labels', [])
        
        try:
            # Verify time dimension matches label length and extract labels
            verified_labels, is_valid = verify_and_extract_labels(
                embedding, parsed_labels, uttid, verbose=verbose
            )
            
            if not is_valid:
                # ipdb.set_trace()
                mismatched_uttids.append(uttid)
                continue  # Skip this utterance if time doesn't match
            
            # Convert numpy arrays to lists for PyArrow compatibility
            # PyArrow can handle nested lists (variable-length list of fixed-size lists)
            embedding_list = embedding.tolist()  # Convert [time, dim] to list of lists
            label_array_list = verified_labels.tolist()  # Convert 1D array to list
            record = {
                'uttid': uttid,
                'embedding': embedding_list,  # List of lists [[dim1, dim2, ...], ...]
                'targets': {
                    'raw': targets['raw'],
                    'labels': targets['labels'],
                    'label_array': label_array_list  # List [0, 1, 0, 1, ...]
                }
            }
            records.append(record)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to process {uttid}: {e}")
            continue
    # ipdb.set_trace()
    # Print mismatched uttids summary
    if mismatched_uttids and verbose:
        print(f"  Found {len(mismatched_uttids)} utterances with time mismatches:")
        for uttid in mismatched_uttids[:10]:  # Print first 10
            print(f"    {uttid}")
        if len(mismatched_uttids) > 10:
            print(f"    ... and {len(mismatched_uttids) - 10} more")
    
    if not records:
        return output_path, 0, 0
    
    # Create DataFrame and save to parquet
    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    file_size = os.path.getsize(output_path) / (1024 ** 2)
    
    return output_path, file_size, len(records)


def archive_to_parquet(
    embeddings_path: str,
    labels_path: str,
    data_name: str,
    output_path: str,
    samples_per_file: int = 1000,
    verbose: bool = True,
    label_reformed: str = "-1",
    add_rir: bool = False
):
    """
    Archive embedding data from joint_embeddings.pt and labels.txt to parquet format.
    
    Args:
        embeddings_path: Path to joint_embeddings.pt file
        labels_path: Path to labels.txt file
        data_name: Name of the dataset (used in output path)
        output_path: Base output directory
        samples_per_file: Number of samples per parquet file
        verbose: Whether to print progress
        label_reformed: Label to re-form the labels.txt file (default: -1, unchanged). 
                        0: Clean, 1: Noisy, 2: RIR, 3: Interference, 4: Packet Loss, 5: Missing, 
                        6: RIR+Noise, 7: RIR+Interference
        add_rir: Add RIR to the final labels (default: False)
    """
    # Load embeddings
    uttid_to_embeddings = load_joint_embeddings(embeddings_path)
    
    if verbose:
        print(f"Parsing labels.txt: {labels_path}")
    uttid_to_targets = parse_labels_txt(labels_path, label_reformed, add_rir)
    
    if verbose:
        print(f"  Embedding entries: {len(uttid_to_embeddings)}")
        print(f"  labels.txt entries: {len(uttid_to_targets)}")
    
    # Find common uttids (intersection)
    common_uttids = sorted(set(uttid_to_embeddings.keys()) & set(uttid_to_targets.keys()))
    
    if verbose:
        print(f"  Common uttids: {len(common_uttids)}")
    
    if not common_uttids:
        print("Error: No common uttids found between embeddings and labels.txt")
        return
    
    # Create output directory: $output_path/$data_name/
    output_dir = os.path.join(output_path, data_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\nOutput directory: {output_dir}")
    
    # ipdb.set_trace()
    # Calculate number of batches
    total_files = len(common_uttids)
    num_batches = (total_files + samples_per_file - 1) // samples_per_file
    
    if verbose:
        print(f"Total utterances: {total_files}")
        print(f"Samples per file: {samples_per_file}")
        print(f"Number of output files: {num_batches}")
    
    # Process in batches
    saved_files = []
    total_size = 0
    total_records = 0
    
    iterator = range(num_batches)
    if verbose:
        iterator = tqdm(iterator, desc="Processing batches")
    
    for batch_idx in iterator:
        start_idx = batch_idx * samples_per_file
        end_idx = min(start_idx + samples_per_file, total_files)
        
        batch_uttids = common_uttids[start_idx:end_idx]
        
        # Generate output filename: ${data_name}_00001.parquet
        output_filename = f"{data_name}_{batch_idx + 1:05d}.parquet"
        batch_output_path = os.path.join(output_dir, output_filename)
        # ipdb.set_trace()
        # Process and save this batch
        output_file, file_size, num_records = process_and_save_batch(
            batch_uttids, uttid_to_embeddings, uttid_to_targets, batch_output_path, verbose=False
        )
        
        saved_files.append(output_file)
        total_size += file_size
        total_records += num_records
    
    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Output directory: {output_dir}")
        print(f"  Total files created: {len(saved_files)}")
        print(f"  Total records: {total_records}")
        print(f"  Total size: {total_size:.2f} MB")
        print(f"  File format: {data_name}_XXXXX.parquet")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Archive embedding data from joint_embeddings.pt and labels.txt to parquet format'
    )
    parser.add_argument(
        '--embeddings', type=str, required=True,
        help='Path to joint_embeddings.pt file (uttid to embedding matrix mapping)'
    )
    parser.add_argument(
        '--labels', type=str, required=True,
        help='Path to labels.txt file (uttid to labels mapping)'
    )
    parser.add_argument(
        '--label_reformed', type=str, default="-1",
        help='Label to re-form the labels.txt file (default: -1, unchanged). 0: Clean, 1: Noisy, 2: RIR, 3: Interference, 4: Packet Loss, 5: Missing, 6: RIR+Noise, 7: RIR+Interference)'
    )
    parser.add_argument(
        '--add_rir', action='store_true', default=False,
        help='Add RIR to the final labels (default: False)'
    )
    parser.add_argument(
        '--data_name', type=str, required=True,
        help='Name of the dataset (used in output path and filenames)'
    )
    parser.add_argument(
        '--output_path', type=str, required=True,
        help='Base output directory (output will be in $output_path/$data_name/)'
    )
    parser.add_argument(
        '--samples_per_file', type=int, default=1000,
        help='Number of samples per parquet file (default: 1000)'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress verbose output'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Validate input files exist
    if not os.path.exists(args.embeddings):
        print(f"Error: embeddings file not found: {args.embeddings}")
        return
    
    if not os.path.exists(args.labels):
        print(f"Error: labels file not found: {args.labels}")
        return
    
    # Archive to parquet
    archive_to_parquet(
        embeddings_path=args.embeddings,
        labels_path=args.labels,
        data_name=args.data_name,
        output_path=args.output_path,
        samples_per_file=args.samples_per_file,
        verbose=verbose,
        label_reformed=args.label_reformed,
        add_rir=args.add_rir
    )
    
    if verbose:
        print("\nDone!")


if __name__ == '__main__':
    main()
