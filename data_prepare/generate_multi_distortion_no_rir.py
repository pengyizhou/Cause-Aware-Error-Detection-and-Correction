"""
Generate audio with multiple non-overlapping distortions (NO RIR).

Each utterance will have:
- 2 to 4 distortion types (packet_loss, missing, noise, interference)
- Each distortion type can occur 1-2 times
- Distortions do not overlap
- NO RIR applied

Usage:
    python generate_multi_distortion_no_rir.py \
        --wav_scp clean_wav.scp --segments clean_segments \
        --noise_wav_scp noise_wav.scp --noise_segments noise_segments \
        --interference_wav_scp interf_wav.scp --interference_segments interf_segments \
        --output_dir output/multi_distortion_no_rir
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
import random

from distortion_augment import (
    mix_background_noise,
    apply_packet_loss_augmentation,
    zero_segment,
    sample_snr,
    generate_frame_labels,
)


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
    if wav_path.strip().endswith('|'):
        cmd = wav_path.strip()[:-1].strip()
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd += f" | sox -t wav - -t wav - trim {start_time} {duration}"
        
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
        audio, sr = sf.read(wav_path)
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


def load_resource_pool(
    wav_scp_path: str,
    segments_path: Optional[str] = None,
    target_sr: int = 16000
) -> List[Dict]:
    """Load resource pool from wav.scp and segments."""
    wav_dict = read_wav_scp(wav_scp_path)
    
    if segments_path and os.path.exists(segments_path):
        segments = read_segments(segments_path)
        resources = []
        
        for seg in segments:
            rec_id = seg['recording_id']
            if rec_id not in wav_dict:
                continue
            
            try:
                audio, sr = load_audio_segment(
                    wav_dict[rec_id],
                    seg['start'],
                    seg['end'],
                    target_sr
                )
                resources.append({
                    'audio': audio,
                    'segment_id': seg['segment_id'],
                    'recording_id': rec_id,
                    'start': seg['start'],
                    'end': seg['end']
                })
            except Exception as e:
                print(f"Warning: Failed to load segment {seg['segment_id']}: {e}")
                continue
        
        return resources
    else:
        resources = []
        for rec_id, wav_path in wav_dict.items():
            try:
                audio, sr = load_audio_segment(wav_path, target_sr=target_sr)
                resources.append({
                    'audio': audio,
                    'recording_id': rec_id
                })
            except Exception as e:
                print(f"Warning: Failed to load recording {rec_id}: {e}")
                continue
        
        return resources


def calculate_num_frames(audio_length: int, frame_hop_s: float, sample_rate: int) -> int:
    """Calculate number of frames for given audio length."""
    frame_hop_samples = int(frame_hop_s * sample_rate)
    return (audio_length - 1) // frame_hop_samples + 1


def generate_non_overlapping_segments(
    duration: float,
    distortion_types: List[str],
    occurrences_per_type: Dict[str, int],
    min_segment_duration: float = 0.2,
    max_segment_duration: float = 1.5,
    min_gap: float = 0.1
) -> List[Tuple[str, float, float]]:
    """
    Generate non-overlapping time segments for distortions.
    
    Args:
        duration: Total audio duration in seconds
        distortion_types: List of distortion types to apply
        occurrences_per_type: Dict mapping distortion type to number of occurrences
        min_segment_duration: Minimum segment duration
        max_segment_duration: Maximum segment duration
        min_gap: Minimum gap between segments
        
    Returns:
        List of (distortion_type, start_time, end_time) tuples
    """
    segments = []
    
    # Create all segments we need
    for dist_type in distortion_types:
        num_occurrences = occurrences_per_type[dist_type]
        for _ in range(num_occurrences):
            seg_duration = random.uniform(min_segment_duration, max_segment_duration)
            segments.append((dist_type, seg_duration))
    
    # Sort by duration (larger first) for better packing
    segments.sort(key=lambda x: x[1], reverse=True)
    
    # Try to place segments without overlap
    placed_segments = []
    max_attempts = 100
    
    for dist_type, seg_duration in segments:
        seg_duration = min(seg_duration, duration * 0.3)  # Don't exceed 30% of total duration
        
        for attempt in range(max_attempts):
            # Random start time
            max_start = duration - seg_duration
            if max_start <= 0:
                break
            
            start_time = random.uniform(0, max_start)
            end_time = start_time + seg_duration
            
            # Check for overlap with existing segments
            overlap = False
            for _, placed_start, placed_end in placed_segments:
                # Check if there's overlap (with min_gap buffer)
                if not (end_time + min_gap <= placed_start or start_time >= placed_end + min_gap):
                    overlap = True
                    break
            
            if not overlap:
                placed_segments.append((dist_type, start_time, end_time))
                break
    
    # Sort by start time
    placed_segments.sort(key=lambda x: x[1])
    
    return placed_segments


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio with multiple non-overlapping distortions (NO RIR)"
    )
    
    # Input/Output
    parser.add_argument('--wav_scp', type=str, required=True,
                       help='Kaldi wav.scp file for clean audio')
    parser.add_argument('--segments', type=str, required=True,
                       help='Kaldi segments file')
    parser.add_argument('--noise_wav_scp', type=str, required=True,
                       help='wav.scp file for noise resources')
    parser.add_argument('--noise_segments', type=str, default=None,
                       help='segments file for noise resources')
    parser.add_argument('--interference_wav_scp', type=str, required=True,
                       help='wav.scp file for interference resources')
    parser.add_argument('--interference_segments', type=str, default=None,
                       help='segments file for interference resources')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    
    # Parameters
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--frame_hop_ms', type=float, default=20.0,
                       help='Frame hop in milliseconds (default: 20 for 50fps)')
    parser.add_argument('--snr_range', type=float, nargs=2, default=[-5, 20],
                       help='SNR range in dB for noise (default: -5 20)')
    parser.add_argument('--sir_range', type=float, nargs=2, default=[5, 20],
                       help='SIR range in dB for interference (default: 5 20)')
    parser.add_argument('--bitrates', type=int, nargs='+', default=[1, 2, 4, 8],
                       help='Bitrates for packet loss (default: 1 2 4 8)')
    parser.add_argument('--min_distortion_types', type=int, default=2,
                       help='Minimum number of distortion types per utterance (default: 2)')
    parser.add_argument('--max_distortion_types', type=int, default=4,
                       help='Maximum number of distortion types per utterance (default: 4)')
    parser.add_argument('--max_segments', type=int, default=None,
                       help='Maximum number of segments to process (for testing)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load clean segments
    print("Loading clean audio segments...")
    wav_scp = read_wav_scp(args.wav_scp)
    segments = read_segments(args.segments)
    
    if args.max_segments:
        segments = segments[:args.max_segments]
    
    print(f"Loaded {len(segments)} segments")
    
    # Load resources
    print("Loading noise resources...")
    noise_resources = load_resource_pool(args.noise_wav_scp, args.noise_segments, args.sample_rate)
    print(f"Loaded {len(noise_resources)} noise resources")
    
    print("Loading interference resources...")
    interference_resources = load_resource_pool(
        args.interference_wav_scp, args.interference_segments, args.sample_rate
    )
    print(f"Loaded {len(interference_resources)} interference resources")
    
    if not noise_resources or not interference_resources:
        print("Error: Missing required resources!")
        return
    
    print(f"\nOutput directory: {output_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Frame rate: {1000/args.frame_hop_ms:.1f} fps ({args.frame_hop_ms}ms hop)")
    print(f"Distortion types per utterance: {args.min_distortion_types}-{args.max_distortion_types}")
    print(f"Each distortion type: 1-2 occurrences")
    print(f"NO RIR applied\n")
    
    # Process segments
    frame_hop_s = args.frame_hop_ms / 1000.0
    success_count = 0
    
    for segment in tqdm(segments, desc="Processing multi-distortion audio (NO RIR)"):
        segment_id = segment['segment_id']
        rec_id = segment['recording_id']
        
        if rec_id not in wav_scp:
            print(f"\nWarning: Recording {rec_id} not found in wav.scp")
            continue
        
        try:
            # Load clean audio (NO RIR applied)
            audio, sr = load_audio_segment(
                wav_scp[rec_id],
                segment['start'],
                segment['end'],
                args.sample_rate
            )
            
            duration = len(audio) / args.sample_rate
            
            # Determine which distortion types to use
            all_distortion_types = ['packet_loss', 'missing', 'noise', 'interference']
            num_types = random.randint(args.min_distortion_types, args.max_distortion_types)
            selected_types = random.sample(all_distortion_types, num_types)
            
            # Determine occurrences for each type (1-2 times)
            occurrences = {dt: random.randint(1, 2) for dt in selected_types}
            
            # Generate non-overlapping segments
            distortion_segments = generate_non_overlapping_segments(
                duration, selected_types, occurrences
            )
            
            # Apply distortions
            events = []
            distorted = audio.copy()
            
            for dist_type, start_s, end_s in distortion_segments:
                start_samp = int(start_s * args.sample_rate)
                end_samp = int(end_s * args.sample_rate)
                
                if dist_type == 'noise':
                    # Apply noise to segment
                    noise_resource = random.choice(noise_resources)
                    noise = noise_resource['audio']
                    snr_db = sample_snr("triangular", tuple(args.snr_range))
                    
                    segment_audio = distorted[start_samp:end_samp]
                    onset = random.randint(0, max(1, len(noise) - len(segment_audio)))
                    mixed_segment = mix_background_noise(segment_audio, noise, snr_db, onset)
                    distorted[start_samp:end_samp] = mixed_segment
                    
                    events.append({
                        "type": "noise",
                        "label": 1,
                        "start": start_s,
                        "end": end_s,
                        "snr_db": snr_db,
                    })
                
                elif dist_type == 'interference':
                    # Apply interference to segment
                    interf_resource = random.choice(interference_resources)
                    interference = interf_resource['audio']
                    sir_db = sample_snr("uniform", tuple(args.sir_range))
                    
                    segment_audio = distorted[start_samp:end_samp]
                    onset = random.randint(0, max(1, len(interference) - len(segment_audio)))
                    mixed_segment = mix_background_noise(segment_audio, interference, sir_db, onset)
                    distorted[start_samp:end_samp] = mixed_segment
                    
                    events.append({
                        "type": "interference",
                        "label": 3,
                        "start": start_s,
                        "end": end_s,
                        "sir_db": sir_db,
                    })
                
                elif dist_type == 'packet_loss':
                    # Apply packet loss to segment
                    bitrate = random.choice(args.bitrates)
                    distorted, event = apply_packet_loss_augmentation(
                        distorted, args.sample_rate, (start_s, end_s), bitrate
                    )
                    events.append(event)
                
                elif dist_type == 'missing':
                    # Apply missing segment
                    distorted, event = zero_segment(
                        distorted, args.sample_rate, (start_s, end_s), fade_ms=5.0
                    )
                    events.append(event)
            
            # Generate frame labels
            num_frames = calculate_num_frames(len(distorted), frame_hop_s, args.sample_rate)
            labels = generate_frame_labels(events, num_frames, frame_hop_s)
            
            # Save audio
            audio_path = output_dir / f"{segment_id}.wav"
            sf.write(str(audio_path), distorted, args.sample_rate)
            
            # Save labels
            labels_path = output_dir / f"{segment_id}.npy"
            np.save(str(labels_path), labels)
            
            # Save metadata
            metadata = {
                'segment_id': segment_id,
                'recording_id': rec_id,
                'start': segment['start'],
                'end': segment['end'],
                'distortion_type': 'multi_distortion_no_rir',
                'rir_applied': False,
                'distortion_types_used': selected_types,
                'distortion_events': events,
                'num_distortion_segments': len(distortion_segments),
                'duration_seconds': len(distorted) / args.sample_rate,
                'num_frames': num_frames
            }
            metadata_path = output_dir / f"{segment_id}.json"
            with open(str(metadata_path), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing segment {segment_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Successfully processed: {success_count}/{len(segments)} segments")
    print(f"Output directory: {output_dir}")
    print("\nEach segment has:")
    print("  - .wav file (audio with multiple distortions, NO RIR)")
    print("  - .npy file (frame-level labels)")
    print("  - .json file (metadata)")
    print("="*60)


if __name__ == "__main__":
    main()
