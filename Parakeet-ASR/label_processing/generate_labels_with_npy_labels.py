#!/usr/bin/env python3
"""
Script to generate frame-level labels from .npy label files.

This version directly loads .npy labels and generates output labels
from timestep 0 to the end of the audio.

Frame size conversion:
    - Input .npy labels: 20ms per frame
    - Output labels: 80ms per frame
    - Conversion ratio: 4 (every 4 input frames → 1 output frame)

Output format:
    uttid\t[frame:label,frame:label,...]
    where frame is the 80ms frame index (0, 1, 2, ...)
"""

import argparse
import math
import os
import sys
import numpy as np
from typing import Optional
from glob import glob


# Frame size in milliseconds
INPUT_FRAME_MS = 20   # npy label frame size
OUTPUT_FRAME_MS = 80  # output label frame size

# Conversion factor: how many input frames per output frame
# 80ms / 20ms = 4
FRAME_RATIO = OUTPUT_FRAME_MS // INPUT_FRAME_MS


def load_npy_labels(npy_path: str) -> Optional[np.ndarray]:
    """Load the .npy label file.
    
    Args:
        npy_path: Path to the .npy file
    
    Returns:
        numpy array of frame-level labels, or None if file not found
    """
    if os.path.exists(npy_path):
        return np.load(npy_path)
    return None


def downsample_labels(npy_labels: np.ndarray, method: str = 'first') -> np.ndarray:
    """Downsample labels from 20ms resolution to 80ms resolution.
    
    Args:
        npy_labels: numpy array of frame-level labels at 20ms resolution
        method: downsampling method
            - 'first': take the first frame of each window (default)
            - 'max': take the maximum label in each window
            - 'majority': take the majority vote in each window
    
    Returns:
        numpy array of labels at 80ms resolution
    """
    num_input_frames = len(npy_labels)
    num_output_frames = math.ceil(num_input_frames / FRAME_RATIO)  # ceiling division
    
    output_labels = []
    
    for i in range(num_output_frames):
        start_idx = i * FRAME_RATIO
        end_idx = min(start_idx + FRAME_RATIO, num_input_frames)
        window = npy_labels[start_idx:end_idx]
        
        if method == 'first':
            # Take the first frame of the window
            label = int(window[0])
        elif method == 'max':
            # Take the maximum label in the window
            label = int(np.max(window))
        elif method == 'majority':
            # Take the majority vote (most common label)
            unique, counts = np.unique(window, return_counts=True)
            label = int(unique[np.argmax(counts)])
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
        
        output_labels.append(label)
    
    return np.array(output_labels, dtype=np.int32)


def process_npy_files(label_dir: str, output_file: str, method: str = 'first'):
    """Process all .npy files in the directory and generate output labels.
    
    Args:
        label_dir: Directory containing .npy label files
        output_file: Path to output file
        method: downsampling method ('first', 'max', or 'majority')
    """
    # Find all .npy files
    npy_pattern = os.path.join(label_dir, "*.npy")
    npy_files = sorted(glob(npy_pattern))
    
    if not npy_files:
        print(f"Error: No .npy files found in {label_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(npy_files)} .npy files in {label_dir}", file=sys.stderr)
    print(f"Downsampling method: {method}", file=sys.stderr)
    print(f"Input frame size: {INPUT_FRAME_MS}ms, Output frame size: {OUTPUT_FRAME_MS}ms", file=sys.stderr)
    
    processed_count = 0
    error_count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out:
        for npy_path in npy_files:
            # Extract uttid from filename (without .npy extension)
            uttid = os.path.splitext(os.path.basename(npy_path))[0]
            
            try:
                # Load npy labels
                npy_labels = load_npy_labels(npy_path)
                
                if npy_labels is None or len(npy_labels) == 0:
                    print(f"Warning: Empty or missing labels for {uttid}", file=sys.stderr)
                    error_count += 1
                    continue
                
                # Downsample from 20ms to 80ms
                output_labels = downsample_labels(npy_labels, method=method)
                
                # Generate output: frame:label for every frame from 0 to end
                events = []
                for frame_idx, label in enumerate(output_labels):
                    events.append(f"{frame_idx}:{label}")
                
                # Format output: uttid [frame:label,frame:label,...]
                output_line = f"{uttid}\t[{','.join(events)}]"
                print(output_line, file=out)
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {uttid}: {e}", file=sys.stderr)
                error_count += 1
    
    print(f"\nProcessed {processed_count} files successfully", file=sys.stderr)
    if error_count > 0:
        print(f"Errors: {error_count} files", file=sys.stderr)
    print(f"Output written to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Generate frame-level labels from .npy label files. '
                    'Converts from 20ms (npy) to 80ms (output) frame resolution.'
    )
    parser.add_argument(
        'label_dir',
        help='Directory containing .npy label files (one per uttid). '
             'Each file should be named <uttid>.npy and contain frame-level labels at 20ms resolution.'
    )
    parser.add_argument(
        'output_file',
        help='Path to output file'
    )
    parser.add_argument(
        '--method',
        choices=['first', 'max', 'majority'],
        default='majority',
        help='Downsampling method: '
             '"first" takes the first frame of each 4-frame window (default), '
             '"max" takes the maximum label, '
             '"majority" takes the most common label'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.label_dir):
        print(f"Error: {args.label_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    process_npy_files(args.label_dir, args.output_file, args.method)


if __name__ == '__main__':
    main()
