"""
Generate clean audio data with zero labels from Kaldi-style wav.scp and segments files.

Usage:
    python generate_clean.py --wav_scp clean_wav.scp --segments clean_segments \
                             --output_dir output/clean --sample_rate 16000
"""

import os
import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import subprocess

from distortion_augment import generate_frame_labels


def read_wav_scp(wav_scp_path: str) -> Dict[str, str]:
    """Read Kaldi wav.scp file."""
    wav_dict = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                rec_id, audio_path = parts
                wav_dict[rec_id] = audio_path
    return wav_dict


def read_segments(segments_path: str) -> List[Dict]:
    """Read Kaldi segments file."""
    segments = []
    with open(segments_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4:
                seg_id, rec_id, start, end = parts[:4]
                segments.append({
                    'segment_id': seg_id,
                    'recording_id': rec_id,
                    'start': float(start),
                    'end': float(end)
                })
    return segments


def load_audio_segment(
    wav_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    target_sr: int = 16000
) -> Tuple[np.ndarray, int]:
    """Load audio segment from file, supporting sox pipe commands."""
    # Check if it's a pipe command
    if wav_path.strip().endswith('|'):
        # Handle sox pipe command
        cmd = wav_path.strip()[:-1].strip()  # Remove trailing |
        
        # Add trim if we have start/end times
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd += f" | sox -t wav - -t wav - trim {start_time} {duration}"
        
        # Execute command and read audio
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        audio, sr = sf.read(process.stdout)
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"Failed to execute: {cmd}")
    else:
        # Regular file path
        audio, sr = sf.read(wav_path)
        
        # Extract segment if times specified
        if start_time is not None and end_time is not None:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio = audio[start_sample:end_sample]
    
    # Resample if needed
    if sr != target_sr:
        from scipy import signal
        num_samples = int(len(audio) * target_sr / sr)
        audio = signal.resample(audio, num_samples)
        sr = target_sr
    
    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio[:, 0]
    
    return audio, sr


def calculate_num_frames(audio_length: int, frame_hop_s: float, sample_rate: int) -> int:
    """Calculate number of frames for given audio length."""
    frame_hop_samples = int(frame_hop_s * sample_rate)
    return (audio_length - 1) // frame_hop_samples + 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate clean audio with zero labels"
    )
    
    # Input/Output
    parser.add_argument('--wav_scp', type=str, required=True,
                       help='Kaldi wav.scp file for clean audio')
    parser.add_argument('--segments', type=str, required=True,
                       help='Kaldi segments file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for clean data')
    
    # Parameters
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--frame_hop_ms', type=float, default=20.0,
                       help='Frame hop in milliseconds (default: 20 for 50fps)')
    parser.add_argument('--max_segments', type=int, default=None,
                       help='Maximum number of segments to process (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load segments
    print("Loading segments...")
    wav_scp = read_wav_scp(args.wav_scp)
    segments = read_segments(args.segments)
    
    if args.max_segments:
        segments = segments[:args.max_segments]
    
    print(f"Loaded {len(segments)} segments from {len(wav_scp)} recordings")
    print(f"Output directory: {output_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Frame rate: {1000/args.frame_hop_ms:.1f} fps ({args.frame_hop_ms}ms hop)\n")
    
    # Process segments
    frame_hop_s = args.frame_hop_ms / 1000.0
    success_count = 0
    
    for segment in tqdm(segments, desc="Processing clean audio"):
        segment_id = segment['segment_id']
        rec_id = segment['recording_id']
        
        if rec_id not in wav_scp:
            print(f"\nWarning: Recording {rec_id} not found in wav.scp")
            continue
        
        try:
            # Load audio segment
            audio, sr = load_audio_segment(
                wav_scp[rec_id],
                segment['start'],
                segment['end'],
                args.sample_rate
            )
            
            # Generate labels (all zeros for clean)
            num_frames = calculate_num_frames(len(audio), frame_hop_s, args.sample_rate)
            labels = np.zeros(num_frames, dtype=np.int64)
            
            # Save audio
            audio_path = output_dir / f"{segment_id}.wav"
            sf.write(str(audio_path), audio, args.sample_rate)
            
            # Save labels
            labels_path = output_dir / f"{segment_id}.npy"
            np.save(str(labels_path), labels)
            
            # Save metadata
            metadata = {
                'segment_id': segment_id,
                'recording_id': rec_id,
                'start': segment['start'],
                'end': segment['end'],
                'distortion_type': 'clean',
                'label': 0,
                'duration_seconds': len(audio) / args.sample_rate,
                'num_frames': num_frames
            }
            metadata_path = output_dir / f"{segment_id}.json"
            with open(str(metadata_path), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing segment {segment_id}: {e}")
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Successfully processed: {success_count}/{len(segments)} segments")
    print(f"Output directory: {output_dir}")
    print("\nEach segment has:")
    print("  - .wav file (clean audio)")
    print("  - .npy file (frame-level labels, all zeros)")
    print("  - .json file (metadata)")
    print("="*60)


if __name__ == "__main__":
    main()
