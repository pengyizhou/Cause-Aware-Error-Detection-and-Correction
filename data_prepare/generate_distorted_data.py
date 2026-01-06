"""
Generate distorted audio data offline using Kaldi-style wav.scp and segments files.

This script processes clean audio segments and generates distorted versions
with corresponding frame-level labels. Each distortion type is saved in
a separate folder.

Input format (Kaldi-style):
    wav.scp: recording_id /path/to/audio.wav
    segments: segment_id recording_id start_time end_time

Output structure:
    output_dir/
    ├── clean/
    │   ├── segment1.wav
    │   ├── segment1.npy (frame labels, all zeros)
    │   └── ...
    ├── noise/
    │   ├── segment1.wav (distorted)
    │   ├── segment1.npy (frame labels)
    │   └── ...
    ├── rir/
    ├── interference/
    ├── packet_loss/
    └── missing/

Label format: numpy array of shape (num_frames,) with values 0-5
"""

import os
import argparse
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
import random
import subprocess
import ipdb

from distortion_augment import (
    apply_rir,
    mix_background_noise, 
    sample_snr,
    apply_interference_speaker_augmentation,
    apply_packet_loss_augmentation,
    zero_segment,
    generate_frame_labels,
)


def read_wav_scp(wav_scp_path: str) -> Dict[str, str]:
    """
    Read Kaldi wav.scp file.
    
    Format: recording_id /path/to/audio.wav
    or: recording_id sox ... |
    
    Args:
        wav_scp_path: Path to wav.scp file
        
    Returns:
        Dictionary mapping recording_id to audio path/command
    """
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
    """
    Read Kaldi segments file.
    
    Format: segment_id recording_id start_time end_time
    
    Args:
        segments_path: Path to segments file
        
    Returns:
        List of segment dictionaries
    """
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
    """
    Load audio segment from file, supporting sox pipe commands.
    
    Args:
        wav_path: Path to audio file or sox command ending with |
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        target_sr: Target sample rate
        
    Returns:
        (audio_array, sample_rate)
    """
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


def load_resource_pool(
    wav_scp_path: str,
    segments_path: Optional[str] = None,
    target_sr: int = 16000
) -> List[Dict]:
    """
    Load resource pool (noise, RIR, or interference) from wav.scp and segments.
    
    Args:
        wav_scp_path: Path to wav.scp file
        segments_path: Path to segments file (optional, if None use full recordings)
        target_sr: Target sample rate
        
    Returns:
        List of resource dictionaries with 'audio' and 'metadata'
    """
    wav_dict = read_wav_scp(wav_scp_path)
    
    if segments_path and os.path.exists(segments_path):
        # Use segments
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
        # Use full recordings
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


class OfflineDistortionGenerator:
    """Generate distorted audio and labels offline."""
    
    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        frame_hop_ms: float = 20.0,
        noise_resources: Optional[List[Dict]] = None,
        rir_resources: Optional[List[Dict]] = None,
        interference_resources: Optional[List[Dict]] = None,
        snr_range: Tuple[float, float] = (-5.0, 20.0),
        sir_range: Tuple[float, float] = (-5.0, 10.0),
        packet_loss_bitrates: List[int] = [6, 8, 12, 16]
    ):
        """
        Args:
            output_dir: Root output directory
            sample_rate: Audio sample rate
            frame_hop_ms: Frame hop in milliseconds (default 20ms for 50fps)
            noise_resources: List of noise audio resources
            rir_resources: List of RIR audio resources
            interference_resources: List of interference speech resources
            snr_range: SNR range for noise augmentation
            sir_range: SIR range for interference augmentation
            packet_loss_bitrates: Bitrate options for packet loss simulation
        """
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.frame_hop_s = frame_hop_ms / 1000.0
        self.noise_resources = noise_resources or []
        self.rir_resources = rir_resources or []
        self.interference_resources = interference_resources or []
        self.snr_range = snr_range
        self.sir_range = sir_range
        self.packet_loss_bitrates = packet_loss_bitrates
        
        # Create output directories
        self.distortion_types = ['clean', 'noise', 'rir', 'interference', 'packet_loss', 'missing']
        for dist_type in self.distortion_types:
            (self.output_dir / dist_type).mkdir(parents=True, exist_ok=True)
        
        print(f"Output directory: {self.output_dir}")
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Frame rate: {1000/frame_hop_ms:.1f} fps ({frame_hop_ms}ms hop)")
        print(f"Noise resources: {len(self.noise_resources)}")
        print(f"RIR resources: {len(self.rir_resources)}")
        print(f"Interference resources: {len(self.interference_resources)}")
    
    def calculate_num_frames(self, audio_length: int) -> int:
        """Calculate number of frames for given audio length."""
        frame_hop_samples = int(self.frame_hop_s * self.sample_rate)
        return (audio_length - 1) // frame_hop_samples + 1
    
    def save_audio_and_labels(
        self,
        audio: np.ndarray,
        labels: np.ndarray,
        output_path: Path,
        metadata: Optional[Dict] = None
    ):
        """
        Save distorted audio and corresponding labels.
        
        Args:
            audio: Audio array
            labels: Frame labels array
            output_path: Output path (without extension)
            metadata: Optional metadata to save
        """
        # Save audio
        sf.write(str(output_path.with_suffix('.wav')), audio, self.sample_rate)
        
        # Save labels
        np.save(str(output_path.with_suffix('.npy')), labels)
        
        # Save metadata if provided
        if metadata is not None:
            with open(str(output_path.with_suffix('.json')), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def generate_clean(self, audio: np.ndarray, audio_id: str, source_info: Dict) -> Dict:
        """Generate clean version (copy original with all-zero labels)."""
        # Generate labels (all zeros for clean)
        num_frames = self.calculate_num_frames(len(audio))
        labels = np.zeros(num_frames, dtype=np.int64)
        
        # Save
        output_path = self.output_dir / 'clean' / audio_id
        metadata = {
            'source_info': source_info,
            'distortion_type': 'clean',
            'label': 0,
            'duration_seconds': len(audio) / self.sample_rate,
            'num_frames': num_frames
        }
        self.save_audio_and_labels(audio, labels, output_path, metadata)
        
        return {'type': 'clean', 'audio': audio, 'labels': labels}
    
    def generate_noise(self, audio: np.ndarray, audio_id: str, source_info: Dict) -> Dict:
        """Generate noisy version."""
        if not self.noise_resources:
            print(f"Warning: No noise resources provided, skipping noise augmentation for {audio_id}")
            return None
        
        # Select random noise resource
        noise_resource = random.choice(self.noise_resources)
        noise = noise_resource['audio']
        
        # Apply noise using inline SNR mixing (avoid file path dependency)
        
        snr_db = sample_snr("triangular", self.snr_range)
        onset = random.randint(0, max(1, len(noise) - len(audio)))
        distorted = mix_background_noise(audio, noise, snr_db, onset)
        
        # Create event for label generation
        event = {
            "type": "noise",
            "label": 1,
            "start": 0.0,
            "end": len(distorted) / self.sample_rate,
            "snr_db": snr_db,
            "noise_resource": noise_resource.get('segment_id', noise_resource.get('recording_id'))
        }
        
        # Generate labels
        num_frames = self.calculate_num_frames(len(distorted))
        labels = generate_frame_labels([event], num_frames, self.frame_hop_s)
        
        # Save
        output_path = self.output_dir / 'noise' / audio_id
        metadata = {
            'source_info': source_info,
            'distortion_type': 'noise',
            'label': 1,
            'snr_db': snr_db,
            'noise_resource': noise_resource.get('segment_id', noise_resource.get('recording_id')),
            'duration_seconds': len(distorted) / self.sample_rate,
            'num_frames': num_frames
        }
        self.save_audio_and_labels(distorted, labels, output_path, metadata)
        
        return {'type': 'noise', 'audio': distorted, 'labels': labels, 'event': event}
    
    def generate_rir(self, audio: np.ndarray, audio_id: str, source_info: Dict) -> Dict:
        """Generate reverberant version."""
        if not self.rir_resources:
            print(f"Warning: No RIR resources provided, skipping RIR augmentation for {audio_id}")
            return None
        
        # Select random RIR resource
        rir_resource = random.choice(self.rir_resources)
        rir = rir_resource['audio']
        
        # Apply RIR
        distorted = apply_rir(audio, rir, normalize_output=True)
        
        # Create event
        event = {
            "type": "rir",
            "label": 2,
            "start": 0.0,
            "end": len(distorted) / self.sample_rate,
            "rir_resource": rir_resource.get('segment_id', rir_resource.get('recording_id'))
        }
        
        # Generate labels
        num_frames = self.calculate_num_frames(len(distorted))
        labels = generate_frame_labels([event], num_frames, self.frame_hop_s)
        
        return {'type': 'rir', 'audio': distorted, 'labels': labels, 'event': event, 'source_info': source_info}
    
    def generate_interference(self, audio: np.ndarray, audio_id: str, source_info: Dict) -> Dict:
        """Generate version with interference speaker."""
        if not self.interference_resources:
            print(f"Warning: No interference resources provided, skipping interference for {audio_id}")
            return None
        
        # Select random interference resource
        interference_resource = random.choice(self.interference_resources)
        interference = interference_resource['audio']
        
        # Random SIR in specified range
        sir_db = random.uniform(self.sir_range[0], self.sir_range[1])
        
        # Apply interference
        distorted = apply_interference_speaker_augmentation(audio, self.sr, interference, sir_db)
        
        # Create event
        event = {
            "type": "interference",
            "label": 3,
            "start": 0.0,
            "end": len(distorted) / self.sample_rate,
            "sir_db": sir_db,
            "interference_resource": interference_resource.get('segment_id', interference_resource.get('recording_id'))
        }
        
        # Generate labels
        num_frames = self.calculate_num_frames(len(distorted))
        labels = generate_frame_labels([event], num_frames, self.frame_hop_s)
        
        return {'type': 'interference', 'audio': distorted, 'labels': labels, 'event': event, 'source_info': source_info}
    
    def generate_packet_loss(
        self,
        audio: np.ndarray,
        audio_id: str,
        source_info: Dict,
        word_timestamps: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate version with packet loss."""
        duration = len(audio) / self.sample_rate
        
        # Choose random segment to corrupt
        if word_timestamps:
            # Use word boundaries
            word = random.choice(word_timestamps)
            start_s = word['start']
            end_s = word['end']
        else:
            # Random segment (0.2 to 1.0 seconds)
            seg_duration = random.uniform(0.2, 1.0)
            max_start = max(0, duration - seg_duration)
            start_s = random.uniform(0, max_start) if max_start > 0 else 0
            end_s = min(start_s + seg_duration, duration)
        
        # Apply packet loss
        bitrate = random.choice(self.packet_loss_bitrates)
        distorted, event = apply_packet_loss_augmentation(
            audio, self.sample_rate, (start_s, end_s), bitrate
        )
        
        # Generate labels
        num_frames = self.calculate_num_frames(len(distorted))
        labels = generate_frame_labels([event], num_frames, self.frame_hop_s)
        
        return {'type': 'packet_loss', 'audio': distorted, 'labels': labels, 'event': event, 'source_info': source_info}
    
    def generate_missing(
        self,
        audio: np.ndarray,
        audio_id: str,
        source_info: Dict,
        word_timestamps: Optional[List[Dict]] = None
    ) -> Dict:
        """Generate version with missing segments."""
        duration = len(audio) / self.sample_rate
        
        # Choose random segment(s) to zero
        if word_timestamps:
            # Use word boundaries, potentially multiple words
            num_words_to_zero = random.randint(1, min(3, len(word_timestamps)))
            words_to_zero = random.sample(word_timestamps, num_words_to_zero)
            events = []
            distorted = audio.copy()
            
            for word in words_to_zero:
                distorted, event = zero_segment(
                    distorted, self.sample_rate,
                    (word['start'], word['end']),
                    fade_ms=5.0
                )
                events.append(event)
        else:
            # Random segment (0.1 to 0.5 seconds)
            seg_duration = random.uniform(0.1, 0.5)
            max_start = max(0, duration - seg_duration)
            start_s = random.uniform(0, max_start) if max_start > 0 else 0
            end_s = min(start_s + seg_duration, duration)
            
            distorted, event = zero_segment(
                audio, self.sample_rate,
                (start_s, end_s),
                fade_ms=5.0
            )
            events = [event]
        
        # Generate labels
        num_frames = self.calculate_num_frames(len(distorted))
        labels = generate_frame_labels(events, num_frames, self.frame_hop_s)
        
        return {'type': 'missing', 'audio': distorted, 'labels': labels, 'events': events, 'source_info': source_info}
        
        return {'type': 'missing', 'audio': distorted, 'labels': labels, 'events': events}
    
    def process_audio_segment(
        self,
        audio: np.ndarray,
        audio_id: str,
        source_info: Dict,
        word_timestamps: Optional[List[Dict]] = None,
        distortion_types: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Process a single audio segment and generate all distortion types.
        
        Args:
            audio: Audio numpy array
            audio_id: Unique identifier (segment ID)
            source_info: Source information (recording_id, start, end, etc.)
            word_timestamps: Optional word-level timestamps
            distortion_types: List of distortion types to generate (default: all)
            
        Returns:
            Dictionary mapping distortion type to generation result
        """
        if distortion_types is None:
            distortion_types = self.distortion_types
        
        results = {}
        
        for dist_type in distortion_types:
            try:
                if dist_type == 'clean':
                    result = self.generate_clean(audio, audio_id, source_info)
                elif dist_type == 'noise':
                    result = self.generate_noise(audio, audio_id, source_info)
                elif dist_type == 'rir':
                    result = self.generate_rir(audio, audio_id, source_info)
                elif dist_type == 'interference':
                    result = self.generate_interference(audio, audio_id, source_info)
                elif dist_type == 'packet_loss':
                    result = self.generate_packet_loss(audio, audio_id, source_info, word_timestamps)
                elif dist_type == 'missing':
                    result = self.generate_missing(audio, audio_id, source_info, word_timestamps)
                else:
                    print(f"Unknown distortion type: {dist_type}")
                    continue
                
                if result is not None:
                    results[dist_type] = result
                    # Save the result
                    self.save_result(result, audio_id, dist_type)
            except Exception as e:
                print(f"Error generating {dist_type} for {audio_id}: {e}")
                import traceback
                traceback.print_exc()
        
        return results
    
    def save_result(self, result: Dict, audio_id: str, dist_type: str):
        """Save audio, labels, and metadata for a distortion result."""
        output_path = self.output_dir / dist_type / audio_id
        audio = result['audio']
        labels = result['labels']
        source_info = result['source_info']
        
        # Prepare metadata
        metadata = {
            'segment_id': audio_id,
            'distortion_type': dist_type,
            'label': result.get('event', {}).get('label', 0),
            'duration_seconds': len(audio) / self.sample_rate,
            'num_frames': len(labels),
            'source_info': source_info
        }
        
        # Add distortion-specific metadata
        if 'event' in result:
            event = result['event']
            if dist_type == 'noise':
                metadata['snr_db'] = event.get('snr_db')
                metadata['noise_resource'] = event.get('noise_resource')
            elif dist_type == 'rir':
                metadata['rir_resource'] = event.get('rir_resource')
            elif dist_type == 'interference':
                metadata['sir_db'] = event.get('sir_db')
                metadata['interference_resource'] = event.get('interference_resource')
            elif dist_type == 'packet_loss':
                metadata['bitrate_kbps'] = event.get('bitrate_kbps')
                metadata['start_time'] = event.get('start')
                metadata['end_time'] = event.get('end')
        
        if 'events' in result:  # For missing segments
            metadata['segments'] = [(e['start'], e['end']) for e in result['events']]
        
        self.save_audio_and_labels(audio, labels, output_path, metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Generate distorted audio data offline with frame-level labels from wav.scp and segments"
    )
    
    # Input/Output
    parser.add_argument('--wav_scp', type=str, required=True,
                       help='Kaldi wav.scp file for clean audio')
    parser.add_argument('--segments', type=str, required=True,
                       help='Kaldi segments file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for distorted data')
    parser.add_argument('--timestamps_json', type=str, default=None,
                       help='JSON file with word timestamps (optional)')
    
    # Augmentation resources (Kaldi format)
    parser.add_argument('--noise_wav_scp', type=str, default=None,
                       help='wav.scp file for noise resources')
    parser.add_argument('--noise_segments', type=str, default=None,
                       help='segments file for noise resources')
    parser.add_argument('--rir_wav_scp', type=str, default=None,
                       help='wav.scp file for RIR resources')
    parser.add_argument('--rir_segments', type=str, default=None,
                       help='segments file for RIR resources')
    parser.add_argument('--interference_wav_scp', type=str, default=None,
                       help='wav.scp file for interference resources')
    parser.add_argument('--interference_segments', type=str, default=None,
                       help='segments file for interference resources')
    
    # Parameters
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Target sample rate (default: 16000)')
    parser.add_argument('--frame_hop_ms', type=float, default=20.0,
                       help='Frame hop in milliseconds (default: 20 for 50fps)')
    parser.add_argument('--snr_range', type=float, nargs=2, default=[-5, 20],
                       help='SNR range for noise (default: -5 20)')
    parser.add_argument('--sir_range', type=float, nargs=2, default=[-5, 10],
                       help='SIR range for interference (default: -5 10)')
    parser.add_argument('--packet_loss_bitrates', type=int, nargs='+',
                       default=[6, 8, 12, 16],
                       help='Bitrates for packet loss (default: 6 8 12 16)')
    
    # Selection
    parser.add_argument('--distortion_types', type=str, nargs='+',
                       default=['clean', 'noise', 'rir', 'interference', 'packet_loss', 'missing'],
                       choices=['clean', 'noise', 'rir', 'interference', 'packet_loss', 'missing'],
                       help='Distortion types to generate')
    parser.add_argument('--max_segments', type=int, default=None,
                       help='Maximum number of segments to process (for testing)')
    
    args = parser.parse_args()
    
    # Load clean audio segments
    print("Loading clean audio segments...")
    wav_scp = read_wav_scp(args.wav_scp)
    segments = read_segments(args.segments)
    print(f"Loaded {len(segments)} segments from {len(wav_scp)} recordings")
    
    # Load noise resources
    noise_resources = []
    if args.noise_wav_scp:
        print("Loading noise resources...")
        noise_resources = load_resource_pool(
            args.noise_wav_scp, 
            args.noise_segments,
            args.sample_rate
        )
        print(f"Loaded {len(noise_resources)} noise resources")
    
    # Load RIR resources
    rir_resources = []
    if args.rir_wav_scp:
        print("Loading RIR resources...")
        rir_resources = load_resource_pool(
            args.rir_wav_scp, 
            args.rir_segments,
            args.sample_rate
        )
        print(f"Loaded {len(rir_resources)} RIR resources")
    
    # Load interference resources
    interference_resources = []
    if args.interference_wav_scp:
        print("Loading interference resources...")
        interference_resources = load_resource_pool(
            args.interference_wav_scp, 
            args.interference_segments,
            args.sample_rate
        )
        print(f"Loaded {len(interference_resources)} interference resources")
    
    # Load timestamps if provided
    timestamps = {}
    if args.timestamps_json and os.path.exists(args.timestamps_json):
        with open(args.timestamps_json, 'r') as f:
            timestamps = json.load(f)
        print(f"Loaded timestamps for {len(timestamps)} segments")
    
    # Create generator
    generator = OfflineDistortionGenerator(
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        frame_hop_ms=args.frame_hop_ms,
        noise_resources=noise_resources,
        rir_resources=rir_resources,
        interference_resources=interference_resources,
        snr_range=tuple(args.snr_range),
        sir_range=tuple(args.sir_range),
        packet_loss_bitrates=args.packet_loss_bitrates
    )
    
    # Process segments
    # segment_list = list(segment.items() for segment in segments)
    segment_list = segments
    # ipdb.set_trace()
    
    print(f"\nProcessing {len(segment_list)} segments...")
    print(f"Generating distortion types: {args.distortion_types}\n")
    
    # Process segments
    stats = {dt: 0 for dt in args.distortion_types}
    
    for segment in tqdm(segment_list, desc="Processing segments"):
        segment_id = segment['segment_id']
        wav_path = wav_scp.get(segment['recording_id'])
        try:
            # Load audio segment
            audio = load_audio_segment(
                wav_path,
                segment['start'],
                segment['end'],
                args.sample_rate
            )
            
            # Get timestamps for this segment if available
            seg_timestamps = timestamps.get(segment_id)
            
            # Source info for metadata
            source_info = {
                'segment_id': segment_id,
                'recording_id': segment['recording_id'],
                'start': segment['start'],
                'end': segment['end']
            }
            
            # Process segment
            results = generator.process_audio_segment(
                audio,
                segment_id,
                source_info,
                word_timestamps=seg_timestamps,
                distortion_types=args.distortion_types
            )
            
            # Update stats
            for dist_type in results:
                stats[dist_type] += 1
                
        except Exception as e:
            print(f"\nError processing segment {segment_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary
    print("\n" + "="*60)
    print("Generation Complete!")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print("\nGenerated files per distortion type:")
    for dist_type, count in stats.items():
        print(f"  {dist_type:15s}: {count} files")
    print("\nEach distorted audio has:")
    print("  - .wav file (distorted audio)")
    print("  - .npy file (frame-level labels)")
    print("  - .json file (metadata)")
    print("="*60)


if __name__ == "__main__":
    main()
