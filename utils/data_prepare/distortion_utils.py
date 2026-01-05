"""
Distortion simulation utilities for frame-level distortion detection.

Distortion types:
- 0: Clean speech
- 1: Noisy background
- 2: Room Impulse Response (RIR)
- 3: Interference speakers
- 4: Network packet loss
- 5: Totally missing segments
"""

import numpy as np
import torch
import torchaudio
import random
from typing import Tuple, List, Optional
import scipy.signal as signal


class DistortionSimulator:
    """Simulate various types of speech distortions."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_dir: Optional[str] = None,
        rir_dir: Optional[str] = None,
        interference_dir: Optional[str] = None
    ):
        self.sample_rate = sample_rate
        self.noise_dir = noise_dir
        self.rir_dir = rir_dir
        self.interference_dir = interference_dir
        
    def add_noise(
        self,
        audio: np.ndarray,
        snr_db: float = 10.0,
        noise_type: str = 'white'
    ) -> np.ndarray:
        """
        Add background noise to audio.
        
        Args:
            audio: Input audio array
            snr_db: Signal-to-noise ratio in dB
            noise_type: Type of noise ('white', 'pink', 'brown', or path to noise file)
            
        Returns:
            Noisy audio array
        """
        signal_power = np.mean(audio ** 2)
        
        if noise_type == 'white':
            noise = np.random.randn(len(audio))
        elif noise_type == 'pink':
            noise = self._generate_pink_noise(len(audio))
        elif noise_type == 'brown':
            noise = self._generate_brown_noise(len(audio))
        else:
            # Load noise from file
            noise, _ = torchaudio.load(noise_type)
            noise = noise.numpy()[0]
            # Repeat or truncate noise to match audio length
            if len(noise) < len(audio):
                noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))[:len(audio)]
            else:
                start_idx = random.randint(0, len(noise) - len(audio))
                noise = noise[start_idx:start_idx + len(audio)]
        
        noise_power = np.mean(noise ** 2)
        
        # Calculate noise scaling factor
        snr_linear = 10 ** (snr_db / 10)
        scale = np.sqrt(signal_power / (noise_power * snr_linear))
        
        return audio + scale * noise
    
    def apply_rir(
        self,
        audio: np.ndarray,
        rir: Optional[np.ndarray] = None,
        rt60: float = 0.5
    ) -> np.ndarray:
        """
        Apply Room Impulse Response (RIR) to simulate room acoustics.
        
        Args:
            audio: Input audio array
            rir: Pre-computed RIR. If None, generates synthetic RIR
            rt60: Reverberation time (RT60) in seconds
            
        Returns:
            Reverberant audio array
        """
        if rir is None:
            rir = self._generate_synthetic_rir(rt60)
        
        # Convolve audio with RIR
        reverb_audio = signal.fftconvolve(audio, rir, mode='same')
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(reverb_audio))
        if max_val > 0:
            reverb_audio = reverb_audio / max_val * np.max(np.abs(audio))
        
        return reverb_audio
    
    def add_interference(
        self,
        audio: np.ndarray,
        interference: np.ndarray,
        sir_db: float = 0.0
    ) -> np.ndarray:
        """
        Add interference speaker to audio.
        
        Args:
            audio: Input audio array
            interference: Interference audio array
            sir_db: Signal-to-interference ratio in dB
            
        Returns:
            Audio with interference
        """
        # Adjust interference length to match audio
        if len(interference) < len(audio):
            interference = np.tile(interference, int(np.ceil(len(audio) / len(interference))))[:len(audio)]
        else:
            start_idx = random.randint(0, len(interference) - len(audio))
            interference = interference[start_idx:start_idx + len(audio)]
        
        signal_power = np.mean(audio ** 2)
        interference_power = np.mean(interference ** 2)
        
        # Calculate interference scaling factor
        sir_linear = 10 ** (sir_db / 10)
        scale = np.sqrt(signal_power / (interference_power * sir_linear))
        
        return audio + scale * interference
    
    def apply_packet_loss(
        self,
        audio: np.ndarray,
        loss_rate: float = 0.1,
        burst_length: int = 320  # ~20ms at 16kHz
    ) -> np.ndarray:
        """
        Simulate network packet loss by zeroing out random bursts.
        
        Args:
            audio: Input audio array
            loss_rate: Probability of packet loss (0-1)
            burst_length: Length of each lost packet in samples
            
        Returns:
            Audio with packet loss
        """
        distorted_audio = audio.copy()
        num_packets = len(audio) // burst_length
        
        for i in range(num_packets):
            if random.random() < loss_rate:
                start_idx = i * burst_length
                end_idx = min((i + 1) * burst_length, len(audio))
                distorted_audio[start_idx:end_idx] = 0
        
        return distorted_audio
    
    def apply_missing_segments(
        self,
        audio: np.ndarray,
        word_timestamps: Optional[List[Tuple[float, float]]] = None,
        missing_prob: float = 0.2
    ) -> np.ndarray:
        """
        Simulate totally missing segments (word-level or random).
        
        Args:
            audio: Input audio array
            word_timestamps: List of (start_time, end_time) tuples in seconds
            missing_prob: Probability of a segment being missing
            
        Returns:
            Audio with missing segments
        """
        distorted_audio = audio.copy()
        
        if word_timestamps is not None:
            # Apply at word level
            for start_time, end_time in word_timestamps:
                if random.random() < missing_prob:
                    start_idx = int(start_time * self.sample_rate)
                    end_idx = int(end_time * self.sample_rate)
                    distorted_audio[start_idx:end_idx] = 0
        else:
            # Apply to random segments
            segment_length = int(0.5 * self.sample_rate)  # 0.5 second segments
            num_segments = len(audio) // segment_length
            
            for i in range(num_segments):
                if random.random() < missing_prob:
                    start_idx = i * segment_length
                    end_idx = min((i + 1) * segment_length, len(audio))
                    distorted_audio[start_idx:end_idx] = 0
        
        return distorted_audio
    
    def _generate_pink_noise(self, length: int) -> np.ndarray:
        """Generate pink noise (1/f noise)."""
        # Simple approximation using filtering
        white = np.random.randn(length)
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = signal.lfilter(b, a, white)
        return pink / np.max(np.abs(pink))
    
    def _generate_brown_noise(self, length: int) -> np.ndarray:
        """Generate brown noise (1/f^2 noise)."""
        white = np.random.randn(length)
        brown = np.cumsum(white)
        return brown / np.max(np.abs(brown))
    
    def _generate_synthetic_rir(self, rt60: float) -> np.ndarray:
        """
        Generate a synthetic room impulse response.
        
        Args:
            rt60: Reverberation time in seconds
            
        Returns:
            Synthetic RIR array
        """
        rir_length = int(rt60 * self.sample_rate)
        
        # Generate exponentially decaying noise
        decay = np.exp(-6.91 * np.arange(rir_length) / (rt60 * self.sample_rate))
        noise = np.random.randn(rir_length)
        rir = decay * noise
        
        # Add direct path (first sample is strongest)
        rir[0] += 1.0
        
        # Normalize
        rir = rir / np.max(np.abs(rir))
        
        return rir


def apply_distortion_with_labels(
    audio: np.ndarray,
    sample_rate: int = 16000,
    word_timestamps: Optional[List[Tuple[float, float]]] = None,
    distortion_config: Optional[dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random distortions to audio and generate frame-level labels.
    
    This function integrates with distortion_augment.py for realistic augmentations.
    
    Args:
        audio: Input clean audio array
        sample_rate: Sampling rate
        word_timestamps: Optional word-level timestamps for targeted distortions
        distortion_config: Configuration for distortion parameters
        
    Returns:
        Tuple of (distorted_audio, frame_labels)
        - distorted_audio: Audio with applied distortions
        - frame_labels: Array of distortion type labels (0-5) per frame
    """
    try:
        # Try to use the new augmentation module
        from distortion_augment import (
            apply_background_noise_augmentation,
            apply_rir_augmentation,
            apply_interference_speaker_augmentation,
            apply_packet_loss_augmentation,
            zero_segment,
            apply_word_level_augmentations,
            generate_frame_labels
        )
        use_new_augment = True
    except ImportError:
        use_new_augment = False
    
    if distortion_config is None:
        distortion_config = {
            'frame_shift': 20,  # ms
            'apply_noise_prob': 0.3,
            'apply_rir_prob': 0.2,
            'apply_interference_prob': 0.2,
            'apply_packet_loss_prob': 0.15,
            'apply_missing_prob': 0.15,
            'snr_range': (5, 20),
            'sir_range': (-5, 5),
            'rt60_range': (0.2, 0.8),
            'noise_files': [],  # Pool of noise files
            'rir_files': [],    # Pool of RIR files
            'interference_files': [],  # Pool of interference files
        }
    
    # Calculate frame parameters
    frame_shift_samples = int(distortion_config['frame_shift'] * sample_rate / 1000)
    hop_s = distortion_config['frame_shift'] / 1000.0  # in seconds
    num_frames = (len(audio) - 1) // frame_shift_samples + 1
    
    # If new augmentation module available and files provided, use it
    if use_new_augment and word_timestamps is not None and (
        distortion_config.get('noise_files') or 
        distortion_config.get('interference_files')
    ):
        # Convert word timestamps to expected format
        word_ts_dicts = [
            {"word": f"w{i}", "start": start, "end": end}
            for i, (start, end) in enumerate(word_timestamps)
        ]
        
        # Apply word-level augmentations
        distorted_audio, events = apply_word_level_augmentations(
            audio, sample_rate, word_ts_dicts,
            noise_files=distortion_config.get('noise_files'),
            interference_files=distortion_config.get('interference_files'),
            p_noise=distortion_config['apply_noise_prob'],
            p_interference=distortion_config['apply_interference_prob'],
            p_packet_loss=distortion_config['apply_packet_loss_prob'],
            p_missing=distortion_config['apply_missing_prob']
        )
        
        # Optionally add global RIR
        if random.random() < distortion_config['apply_rir_prob'] and distortion_config.get('rir_files'):
            from distortion_augment import apply_rir_augmentation
            distorted_audio, rir_event = apply_rir_augmentation(
                distorted_audio, sample_rate, distortion_config['rir_files']
            )
            events.append(rir_event)
        
        # Generate frame labels from events
        frame_labels = generate_frame_labels(events, num_frames, hop_s)
        
        return distorted_audio, frame_labels
    
    # Fall back to old implementation
    simulator = DistortionSimulator(sample_rate=sample_rate)
    
    # Initialize frame labels (0 = clean)
    frame_labels = np.zeros(num_frames, dtype=np.int64)
    
    distorted_audio = audio.copy()
    
    # Randomly decide which distortions to apply
    distortions = []
    if random.random() < distortion_config['apply_noise_prob']:
        distortions.append(1)
    if random.random() < distortion_config['apply_rir_prob']:
        distortions.append(2)
    if random.random() < distortion_config['apply_interference_prob']:
        distortions.append(3)
    if random.random() < distortion_config['apply_packet_loss_prob']:
        distortions.append(4)
    if random.random() < distortion_config['apply_missing_prob']:
        distortions.append(5)
    
    # Apply distortions
    for dist_type in distortions:
        if dist_type == 1:  # Noise
            snr = random.uniform(*distortion_config['snr_range'])
            distorted_audio = simulator.add_noise(distorted_audio, snr_db=snr)
            # Noise affects all frames
            frame_labels[frame_labels == 0] = 1
            
        elif dist_type == 2:  # RIR
            rt60 = random.uniform(*distortion_config['rt60_range'])
            distorted_audio = simulator.apply_rir(distorted_audio, rt60=rt60)
            # RIR affects all frames
            frame_labels[frame_labels == 0] = 2
            
        elif dist_type == 3:  # Interference
            # Generate synthetic interference (or load from file)
            interference = np.random.randn(len(audio)) * 0.5
            sir = random.uniform(*distortion_config['sir_range'])
            distorted_audio = simulator.add_interference(distorted_audio, interference, sir_db=sir)
            # Interference affects all frames
            frame_labels[frame_labels == 0] = 3
            
        elif dist_type == 4:  # Packet loss
            loss_rate = random.uniform(0.05, 0.2)
            mask = np.ones(len(audio), dtype=bool)
            distorted_audio = simulator.apply_packet_loss(distorted_audio, loss_rate=loss_rate)
            
            # Mark affected frames
            for i in range(len(audio)):
                if distorted_audio[i] == 0 and audio[i] != 0:
                    mask[i] = False
            
            for frame_idx in range(num_frames):
                start_idx = frame_idx * frame_shift_samples
                end_idx = min(start_idx + frame_shift_samples, len(audio))
                if not np.all(mask[start_idx:end_idx]):
                    frame_labels[frame_idx] = 4
                    
        elif dist_type == 5:  # Missing segments
            mask = np.ones(len(audio), dtype=bool)
            distorted_audio = simulator.apply_missing_segments(
                distorted_audio,
                word_timestamps=word_timestamps,
                missing_prob=0.2
            )
            
            # Mark affected frames
            for i in range(len(audio)):
                if distorted_audio[i] == 0 and audio[i] != 0:
                    mask[i] = False
            
            for frame_idx in range(num_frames):
                start_idx = frame_idx * frame_shift_samples
                end_idx = min(start_idx + frame_shift_samples, len(audio))
                if not np.all(mask[start_idx:end_idx]):
                    frame_labels[frame_idx] = 5
    
    return distorted_audio, frame_labels


if __name__ == "__main__":
    # Example usage
    print("Testing distortion simulator...")
    
    # Generate a simple test signal
    duration = 3.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(duration * sample_rate))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Apply distortions
    distorted, labels = apply_distortion_with_labels(audio, sample_rate)
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Distorted audio shape: {distorted.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
    print(f"Label distribution: {[(i, np.sum(labels == i)) for i in range(6)]}")
