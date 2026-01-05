#!/usr/bin/env python3
"""
Archive audio data from wav.scp and labels.txt to parquet format.
Only includes utterances that exist in both files.

Output format: $output_path/$data_name/${data_name}_00001.parquet
Each record contains: uttid, audio (bytes), targets (dict)
"""

import os
import json
import argparse
from shutil import register_unpack_format
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import numpy as np
import soundfile as sf


def parse_wav_scp(wav_scp_path: str) -> Dict[str, str]:
    """
    Parse wav.scp file to get recording_id -> wav_path mapping.
    
    Args:
        wav_scp_path: Path to wav.scp file
        
    Returns:
        Dict mapping recording_id (or uttid) to wav file path
    """
    recording_id_to_path = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by whitespace (could be space or tab)
            parts = line.split()
            if len(parts) >= 2:
                recording_id = parts[0]
                wav_path = parts[1]
                recording_id_to_path[recording_id] = wav_path
    return recording_id_to_path


def parse_segments(segments_path: str) -> Dict[str, Tuple[str, float, float]]:
    """
    Parse segments file to get uttid -> (recording_id, start_time, end_time) mapping.
    Format: uttid recording_id start_time end_time
    
    Args:
        segments_path: Path to segments file
        
    Returns:
        Dict mapping uttid to (recording_id, start_time, end_time) tuple
    """
    uttid_to_segment = {}
    with open(segments_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by whitespace
            parts = line.split()
            if len(parts) >= 4:
                uttid = parts[0]
                recording_id = parts[1]
                start_time = float(parts[2])
                end_time = float(parts[3])
                uttid_to_segment[uttid] = (recording_id, start_time, end_time)
    return uttid_to_segment


def parse_labels_txt(labels_path: str, label_reformed: str = "1", add_rir: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Parse labels.txt file to get uttid -> labels dict mapping.
    Labels format: uttid\t[frame_idx:label,...]
    
    Args:
        labels_path: Path to labels.txt file
        
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
                                # if label == 0:
                                #     parsed_labels.append({
                                #         'frame_idx': int(frame_idx),
                                #         'label': int(label)
                                #     })
                                # elif label == 1:
                                #     parsed_labels.append({
                                #         'frame_idx': int(frame_idx),
                                #         'label': int(label_reformed)
                                #     })
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
                
                uttid_to_targets[uttid] = {
                    'raw': labels_str,
                    'labels': parsed_labels
                }
    return uttid_to_targets


def load_audio_bytes(
    wav_path: str, 
    convert_to_mono: bool = True,
    start_time: float = None,
    end_time: float = None
) -> Tuple[bytes, int]:
    """
    Load audio file and return as np.float32 bytes.
    If stereo, optionally convert to mono by averaging channels.
    Optionally extract a segment using start_time and end_time.
    Uses efficient IO to read only the required segment when start/end times are provided.
    
    Args:
        wav_path: Path to wav file
        convert_to_mono: If True, convert stereo to mono
        start_time: Optional start time in seconds for segment extraction
        end_time: Optional end time in seconds for segment extraction
        
    Returns:
        Tuple of (audio as np.float32 bytes, sample_rate)
    """
    
    # Get file info to determine sample rate and duration (fast, only reads metadata)
    file_info = sf.info(wav_path)
    sr = file_info.samplerate
    total_frames = file_info.frames
    
    # If segment times are provided, read only that portion directly from file
    if start_time is not None and end_time is not None:
        start_frame = int(start_time * sr)
        end_frame = int(end_time * sr)
        # Ensure indices are within bounds
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        
        # Read only the required segment (efficient IO)
        audio = sf.read(wav_path, dtype='float32', start=start_frame, stop=end_frame)[0]
    else:
        # Read entire file
        audio = sf.read(wav_path, dtype='float32')[0]
    
    # Convert stereo to mono if needed
    if convert_to_mono and len(audio.shape) > 1 and audio.shape[1] > 1:
        audio = audio.mean(axis=1).astype(np.float32)
    
    # Return as float32 bytes
    return audio, sr


def process_and_save_batch(
    batch_uttids: List[str],
    uttid_to_path: Dict[str, str],
    uttid_to_targets: Dict[str, Dict],
    output_path: str,
    verbose: bool = True,
    uttid_to_segment: Dict[str, Tuple[str, float, float]] = None
) -> Tuple[str, float, int]:
    """
    Process a batch of utterances and save to parquet.
    
    Args:
        batch_uttids: List of uttids to process
        uttid_to_path: Dict mapping uttid (or recording_id) to wav path
        uttid_to_targets: Dict mapping uttid to targets dict
        output_path: Full output file path
        verbose: Whether to print progress
        uttid_to_segment: Optional dict mapping uttid to (recording_id, start_time, end_time)
        
    Returns:
        Tuple of (output_file_path, file_size_MB, num_records)
    """
    records = []
    
    for uttid in batch_uttids:
        targets = uttid_to_targets[uttid]
        
        # Determine wav path and segment times
        if uttid_to_segment and uttid in uttid_to_segment:
            # Use segments: look up recording_id, then get wav path
            recording_id, start_time, end_time = uttid_to_segment[uttid]
            if recording_id not in uttid_to_path:
                if verbose:
                    print(f"  Warning: Recording ID {recording_id} not found in wav.scp for uttid {uttid}")
                continue
            wav_path = uttid_to_path[recording_id]
            segment_start = start_time
            segment_end = end_time
        else:
            # Direct mapping: uttid -> wav_path
            wav_path = uttid_to_path[uttid]
            segment_start = None
            segment_end = None
        
        try:
            audio, sample_rate = load_audio_bytes(
                wav_path, 
                start_time=segment_start, 
                end_time=segment_end
            )
            
            record = {
                'uttid': uttid,
                'audio': audio,
                'sample_rate': sample_rate,
                'targets': targets  # This is a dict with 'raw' and 'labels' keys
            }
            records.append(record)
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to load {wav_path}: {e}")
            continue
    
    if not records:
        return output_path, 0, 0
    
    # Create DataFrame and save to parquet
    df = pd.DataFrame(records)
    df.to_parquet(output_path, engine='pyarrow', compression='snappy')
    
    file_size = os.path.getsize(output_path) / (1024 ** 2)
    
    return output_path, file_size, len(records)


def archive_to_parquet(
    wav_scp_path: str,
    labels_path: str,
    data_name: str,
    output_path: str,
    samples_per_file: int = 1000,
    verbose: bool = True,
    segments_path: str = None,
    label_reformed: str = "1",
    add_rir: bool = False
):
    """
    Archive audio data from wav.scp and labels.txt to parquet format.
    
    Args:
        wav_scp_path: Path to wav.scp file
        labels_path: Path to labels.txt file
        data_name: Name of the dataset (used in output path)
        output_path: Base output directory
        samples_per_file: Number of samples per parquet file
        verbose: Whether to print progress
        segments_path: Optional path to segments file (uttid recording_id start_time end_time)
        label_reformed: Label to re-form the labels.txt file (default: 1, unchanged). 0: Clean, 1: Noisy, 2: RIR, 3: Interference, 4: Packet Loss, 5: Missing, 6: RIR+Noise, 7: RIR+Interference)
        add_rir: Add RIR to the final labels (default: False)
    """
    # Parse input files
    if verbose:
        print(f"Parsing wav.scp: {wav_scp_path}")
    recording_id_to_path = parse_wav_scp(wav_scp_path)
    
    if verbose:
        print(f"Parsing labels.txt: {labels_path}")
    uttid_to_targets = parse_labels_txt(labels_path, label_reformed, add_rir)
    
    # Parse segments file if provided
    uttid_to_segment = None
    if segments_path:
        if verbose:
            print(f"Parsing segments: {segments_path}")
        uttid_to_segment = parse_segments(segments_path)
        if verbose:
            print(f"  segments entries: {len(uttid_to_segment)}")
    
    if verbose:
        print(f"  wav.scp entries: {len(recording_id_to_path)}")
        print(f"  labels.txt entries: {len(uttid_to_targets)}")
    
    # Determine common uttids
    if segments_path:
        # When segments file is provided, uttids come from segments file
        # and we need to verify recording_ids exist in wav.scp
        segment_uttids = set(uttid_to_segment.keys())
        label_uttids = set(uttid_to_targets.keys())
        common_uttids_set = segment_uttids & label_uttids
        
        # Also verify that all recording_ids from segments exist in wav.scp
        missing_recordings = []
        valid_uttids = set()
        for uttid in common_uttids_set:
            recording_id, _, _ = uttid_to_segment[uttid]
            if recording_id in recording_id_to_path:
                valid_uttids.add(uttid)
            else:
                missing_recordings.append((uttid, recording_id))
        
        if missing_recordings:
            if verbose:
                print(f"  Warning: {len(missing_recordings)} uttids have recording_ids not found in wav.scp")
        
        common_uttids = sorted(valid_uttids)
    else:
        # When segments file is not provided, uttids come from wav.scp
        common_uttids = sorted(set(recording_id_to_path.keys()) & set(uttid_to_targets.keys()))
    
    if verbose:
        print(f"  Common uttids: {len(common_uttids)}")
    
    if not common_uttids:
        print("Error: No common uttids found")
        return
    
    # Create output directory: $output_path/$data_name/
    output_dir = os.path.join(output_path, data_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\nOutput directory: {output_dir}")
    
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
        
        # Process and save this batch
        output_file, file_size, num_records = process_and_save_batch(
            batch_uttids, recording_id_to_path, uttid_to_targets, batch_output_path, 
            verbose=False, uttid_to_segment=uttid_to_segment
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
        description='Archive audio data from wav.scp and labels.txt to parquet format'
    )
    parser.add_argument(
        '--wav_scp', type=str, required=True,
        help='Path to wav.scp file (uttid to wav path mapping)'
    )
    parser.add_argument(
        '--labels', type=str, required=True,
        help='Path to labels.txt file (uttid to labels mapping)'
    )
    parser.add_argument(
        '--label_reformed', type=str, default="-1",
        help='Label to re-form the labels.txt file (default: 1, unchanged). 0: Clean, 1: Noisy, 2: RIR, 3: Interference, 4: Packet Loss, 5: Missing, 6: RIR+Noise, 7: RIR+Interference)'
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
    parser.add_argument(
        '--segments', type=str, default=None,
        help='Optional path to segments file (format: uttid recording_id start_time end_time). '
             'If provided, uttids are taken from segments file and wav paths are looked up from wav.scp using recording_id'
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Validate input files exist
    if not os.path.exists(args.wav_scp):
        print(f"Error: wav.scp file not found: {args.wav_scp}")
        return
    
    if not os.path.exists(args.labels):
        print(f"Error: labels file not found: {args.labels}")
        return
    
    if args.segments and not os.path.exists(args.segments):
        print(f"Error: segments file not found: {args.segments}")
        return
    
    # Archive to parquet
    archive_to_parquet(
        wav_scp_path=args.wav_scp,
        labels_path=args.labels,
        data_name=args.data_name,
        output_path=args.output_path,
        samples_per_file=args.samples_per_file,
        verbose=verbose,
        segments_path=args.segments,
        label_reformed=args.label_reformed,
        add_rir=args.add_rir
    )
    
    if verbose:
        print("\nDone!")


if __name__ == '__main__':
    main()
