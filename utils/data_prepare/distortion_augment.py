"""
Distortion augmentation utilities for frame-level distortion detection.

This module provides functions to simulate various types of speech distortions:
- Noisy background (SNR-based mixing)
- Room Impulse Response (RIR) / Reverberation
- Interference speakers (speech-on-speech)
- Network packet loss / low bitrate codec artifacts
- Missing segments (zeroed audio)

All functions return augmented audio and event metadata for frame labeling.
"""

import numpy as np
import tempfile
import subprocess
import soundfile as sf
import os
from typing import Dict, List, Tuple, Optional, Union
from scipy.signal import fftconvolve
import warnings


# ============================================================================
# Helper functions
# ============================================================================

def rms(x: np.ndarray) -> float:
    """Calculate RMS (Root Mean Square) of a signal."""
    return np.sqrt(np.mean(x**2) + 1e-12)


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize audio to target peak amplitude."""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio * (target_peak / max_val)
    return audio


def scale_for_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale noise signal to achieve target SNR when mixed with clean signal.
    
    SNR (dB) = 20 * log10(RMS_clean / RMS_noise)
    
    Args:
        clean: Clean signal
        noise: Noise signal (will be scaled)
        snr_db: Target signal-to-noise ratio in dB
        
    Returns:
        Scaled noise signal
    """
    r_clean = rms(clean)
    r_noise = rms(noise)
    
    if r_noise == 0:
        return noise
    
    # Calculate target noise RMS for desired SNR
    target_noise_rms = r_clean / (10 ** (snr_db / 20.0))
    scale_factor = target_noise_rms / r_noise
    
    return noise * scale_factor


def sample_snr(distribution: str = "uniform", 
               snr_range: Tuple[float, float] = (-5.0, 20.0)) -> float:
    """
    Sample SNR value from specified distribution.
    
    Args:
        distribution: "uniform" or "triangular"
        snr_range: (min_snr, max_snr) in dB
        
    Returns:
        Sampled SNR value in dB
    """
    min_snr, max_snr = snr_range
    
    if distribution == "uniform":
        return np.random.uniform(min_snr, max_snr)
    elif distribution == "triangular":
        # More probability around middle values
        mode = (min_snr + max_snr) / 2
        return np.random.triangular(min_snr, mode, max_snr)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def apply_fade(audio: np.ndarray, sr: int, fade_ms: float = 5.0, 
               fade_in: bool = True, fade_out: bool = True) -> np.ndarray:
    """
    Apply fade in/out to audio to avoid clicks.
    
    Args:
        audio: Audio signal
        sr: Sample rate
        fade_ms: Fade duration in milliseconds
        fade_in: Whether to apply fade in
        fade_out: Whether to apply fade out
        
    Returns:
        Audio with fades applied
    """
    fade_samples = int(fade_ms * sr / 1000)
    fade_samples = min(fade_samples, len(audio) // 2)
    
    result = audio.copy()
    
    if fade_in and fade_samples > 0:
        fade_curve = np.linspace(0.0, 1.0, fade_samples)
        result[:fade_samples] *= fade_curve
    
    if fade_out and fade_samples > 0:
        fade_curve = np.linspace(1.0, 0.0, fade_samples)
        result[-fade_samples:] *= fade_curve
    
    return result


# ============================================================================
# RIR (Reverberation) Augmentation
# ============================================================================

def apply_rir(audio: np.ndarray, rir: np.ndarray, 
              normalize_output: bool = True) -> np.ndarray:
    """
    Apply Room Impulse Response (RIR) to audio via convolution.
    
    Args:
        audio: Input audio signal
        rir: Room impulse response
        normalize_output: Whether to normalize output to match input RMS
        
    Returns:
        Convolved (reverberated) audio
    """
    # Convolve audio with RIR
    convolved = fftconvolve(audio, rir, mode='full')[:len(audio)]
    
    # Normalize to preserve RMS energy
    if normalize_output:
        orig_rms = rms(audio)
        conv_rms = rms(convolved)
        if conv_rms > 0:
            convolved = convolved * (orig_rms / conv_rms)
    
    # Clip to prevent overflow
    convolved = np.clip(convolved, -1.0, 1.0)
    
    return convolved


def apply_rir_augmentation(audio: np.ndarray, sr: int, 
                          rir_files: List[str]) -> Tuple[np.ndarray, Dict]:
    """
    Apply random RIR from a pool of RIR files.
    
    Args:
        audio: Input audio
        sr: Sample rate
        rir_files: List of paths to RIR files
        
    Returns:
        (augmented_audio, event_dict) where event_dict contains:
            - type: "rir"
            - start: 0.0 (applied to full utterance)
            - end: duration in seconds
            - rir_file: path to RIR used
    """
    if not rir_files:
        raise ValueError("No RIR files provided")
    
    # Load random RIR
    rir_file = np.random.choice(rir_files)
    rir, rir_sr = sf.read(rir_file)
    
    # Resample RIR if needed
    if rir_sr != sr:
        # Simple resampling (for production, use librosa.resample)
        from scipy import signal
        num_samples = int(len(rir) * sr / rir_sr)
        rir = signal.resample(rir, num_samples)
    
    # If stereo, take first channel
    if rir.ndim > 1:
        rir = rir[:, 0]
    
    # Apply RIR
    augmented = apply_rir(audio, rir, normalize_output=True)
    
    event = {
        "type": "rir",
        "label": 2,
        "start": 0.0,
        "end": len(audio) / sr,
        "rir_file": rir_file
    }
    
    return augmented, event


# ============================================================================
# Noisy Background Augmentation
# ============================================================================

def mix_background_noise(audio: np.ndarray, noise: np.ndarray, 
                        snr_db: float, onset_samples: int = None) -> np.ndarray:
    """
    Mix background noise with audio at specified SNR.
    
    Args:
        audio: Clean audio signal
        noise: Noise signal
        snr_db: Target SNR in dB
        onset_samples: Start position in noise to begin mixing (if None, randomly selected 
                       based on available length to avoid tiling)
        
    Returns:
        Mixed audio
    """
    N = len(audio)
    
    # Randomly select onset if not specified, ensuring we have enough noise without tiling
    if onset_samples is None:
        # Calculate the maximum onset position that still leaves enough noise
        max_onset = max(0, len(noise) - N)
        if max_onset > 0:
            onset_samples = np.random.randint(0, max_onset + 1)
        else:
            # Noise is shorter than audio, will need to tile
            onset_samples = 0
    
    # Tile noise only if too short
    if len(noise) - onset_samples < N:
        repeats = int(np.ceil((N + onset_samples) / len(noise)))
        noise = np.tile(noise, repeats)
    
    # Extract noise segment
    noise_seg = noise[onset_samples:onset_samples + N]
    
    # Scale noise to target SNR
    scaled_noise = scale_for_snr(audio, noise_seg, snr_db)
    
    # Mix
    mixed = audio + scaled_noise
    
    # Normalize to prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    
    return mixed


def apply_background_noise_augmentation(
    audio: np.ndarray, sr: int, noise_files: List[str],
    snr_range: Tuple[float, float] = (-5.0, 20.0),
    apply_to_segment: bool = False,
    segment_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Apply background noise to audio (full or partial).
    
    Args:
        audio: Input audio
        sr: Sample rate
        noise_files: List of paths to noise files
        snr_range: (min_snr, max_snr) in dB
        apply_to_segment: Whether to apply to partial segment
        segment_range: (start_s, end_s) if apply_to_segment=True
        
    Returns:
        (augmented_audio, event_dict)
    """
    if not noise_files:
        raise ValueError("No noise files provided")
    
    # Load random noise
    noise_file = np.random.choice(noise_files)
    noise, noise_sr = sf.read(noise_file)
    
    # Resample if needed
    if noise_sr != sr:
        from scipy import signal
        num_samples = int(len(noise) * sr / noise_sr)
        noise = signal.resample(noise, num_samples)
    
    # If stereo, take first channel
    if noise.ndim > 1:
        noise = noise[:, 0]
    
    # Sample SNR
    snr_db = sample_snr("triangular", snr_range)
    
    # Determine segment to augment
    if apply_to_segment and segment_range is not None:
        start_s, end_s = segment_range
        start_samp = int(start_s * sr)
        end_samp = int(end_s * sr)
        
        # Apply to segment
        segment = audio[start_samp:end_samp]
        onset = np.random.randint(0, max(1, len(noise) - len(segment)))
        mixed_segment = mix_background_noise(segment, noise, snr_db, onset)
        
        # Reconstruct audio
        augmented = audio.copy()
        augmented[start_samp:end_samp] = mixed_segment
        
        event = {
            "type": "noise",
            "label": 1,
            "start": start_s,
            "end": end_s,
            "snr_db": snr_db,
            "noise_file": noise_file
        }
    else:
        # Apply to full utterance
        onset = np.random.randint(0, max(1, len(noise) - len(audio)))
        augmented = mix_background_noise(audio, noise, snr_db, onset)
        
        event = {
            "type": "noise",
            "label": 1,
            "start": 0.0,
            "end": len(audio) / sr,
            "snr_db": snr_db,
            "noise_file": noise_file
        }
    
    return augmented, event


# ============================================================================
# Interference Speaker Augmentation
# ============================================================================

def apply_interference_speaker_augmentation(
    audio: np.ndarray, sr: int, interference_files: List[str],
    snr_range: Tuple[float, float] = (-5.0, 10.0),
    apply_to_segment: bool = False,
    segment_range: Optional[Tuple[float, float]] = None
) -> Tuple[np.ndarray, Dict]:
    """
    Add interference speaker (speech-on-speech).
    
    Args:
        audio: Input audio
        sr: Sample rate
        interference_files: List of paths to interference speech files
        snr_range: (min_snr, max_snr) in dB (lower = interference louder)
        apply_to_segment: Whether to apply to partial segment
        segment_range: (start_s, end_s) if apply_to_segment=True
        
    Returns:
        (augmented_audio, event_dict)
    """
    if not interference_files:
        raise ValueError("No interference files provided")
    
    # Load random interference speech
    interf_file = np.random.choice(interference_files)
    interference, interf_sr = sf.read(interf_file)
    
    # Resample if needed
    if interf_sr != sr:
        from scipy import signal
        num_samples = int(len(interference) * sr / interf_sr)
        interference = signal.resample(interference, num_samples)
    
    # If stereo, take first channel
    if interference.ndim > 1:
        interference = interference[:, 0]
    
    # Sample SNR
    snr_db = sample_snr("uniform", snr_range)
    
    # Determine segment
    if apply_to_segment and segment_range is not None:
        start_s, end_s = segment_range
        start_samp = int(start_s * sr)
        end_samp = int(end_s * sr)
        
        segment = audio[start_samp:end_samp]
        onset = np.random.randint(0, max(1, len(interference) - len(segment)))
        mixed_segment = mix_background_noise(segment, interference, snr_db, onset)
        
        augmented = audio.copy()
        augmented[start_samp:end_samp] = mixed_segment
        
        event = {
            "type": "interference",
            "label": 3,
            "start": start_s,
            "end": end_s,
            "snr_db": snr_db,
            "interference_file": interf_file
        }
    else:
        # Full utterance
        onset = np.random.randint(0, max(1, len(interference) - len(audio)))
        augmented = mix_background_noise(audio, interference, snr_db, onset)
        
        event = {
            "type": "interference",
            "label": 3,
            "start": 0.0,
            "end": len(audio) / sr,
            "snr_db": snr_db,
            "interference_file": interf_file
        }
    
    return augmented, event


# ============================================================================
# Network Packet Loss / Low Bitrate Codec Augmentation
# ============================================================================

def reencode_segment_low_bitrate(segment: np.ndarray, sr: int, 
                                 bitrate_kbps: int = 8,
                                 codec: str = "libopus") -> np.ndarray:
    """
    Re-encode audio segment with low bitrate codec to simulate network artifacts.
    
    Args:
        segment: Audio segment
        sr: Sample rate
        bitrate_kbps: Target bitrate in kbps
        codec: Codec to use ("libopus", "libgsm", etc.)
        
    Returns:
        Re-encoded segment
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        in_wav = os.path.join(tmpdir, "in.wav")
        out_wav = os.path.join(tmpdir, "out.wav")
        out_opus = os.path.join(tmpdir, "out.opus")
        
        # Write input
        sf.write(in_wav, segment, sr, subtype='PCM_16')
        
        # Re-encode with ffmpeg
        cmd = [
            "/home/asrxiv/.local/share/ffmpeg-7.0.2-amd64-static/ffmpeg", "-y", "-loglevel", "error",
            "-i", in_wav,
            "-c:a", codec,
            "-b:a", f"{bitrate_kbps}k",
            out_opus
        ]
        
        try:
            subprocess.check_call(cmd, stderr=subprocess.DEVNULL)
            out, _ = sf.read(out_opus, dtype='float32')

            # Match length (may differ slightly)
            if len(out) < len(segment):
                out = np.pad(out, (0, len(segment) - len(out)))
            else:
                out = out[:len(segment)]
            
            return out
        except Exception as e:
            warnings.warn(f"FFmpeg encoding failed: {e}. Returning original segment.")
            return segment


def apply_packet_loss_augmentation(
    audio: np.ndarray, sr: int,
    segment_range: Tuple[float, float],
    bitrate_kbps: int = 8
) -> Tuple[np.ndarray, Dict]:
    """
    Simulate packet loss by re-encoding segment with low bitrate codec.
    
    Args:
        audio: Input audio
        sr: Sample rate
        segment_range: (start_s, end_s) to apply distortion
        bitrate_kbps: Target bitrate (6-16 kbps realistic)
        
    Returns:
        (augmented_audio, event_dict)
    """
    start_s, end_s = segment_range
    start_samp = int(start_s * sr)
    end_samp = int(end_s * sr)
    
    # Extract segment
    segment = audio[start_samp:end_samp]
    
    # Re-encode
    distorted_segment = reencode_segment_low_bitrate(segment, sr, bitrate_kbps)
    
    # Apply fade to avoid harsh transitions
    distorted_segment = apply_fade(distorted_segment, sr, fade_ms=5.0)
    
    # Reconstruct audio
    augmented = audio.copy()
    augmented[start_samp:end_samp] = distorted_segment
    
    event = {
        "type": "packet_loss",
        "label": 4,
        "start": start_s,
        "end": end_s,
        "bitrate_kbps": bitrate_kbps
    }
    
    return augmented, event


# ============================================================================
# Missing Segment Augmentation
# ============================================================================

def zero_segment(audio: np.ndarray, sr: int, 
                segment_range: Tuple[float, float],
                fade_ms: float = 5.0) -> Tuple[np.ndarray, Dict]:
    """
    Zero out (mute) a segment of audio to simulate missing data.
    
    Args:
        audio: Input audio
        sr: Sample rate
        segment_range: (start_s, end_s) to zero out
        fade_ms: Fade duration to avoid clicks
        
    Returns:
        (augmented_audio, event_dict)
    """
    start_s, end_s = segment_range
    start_samp = int(start_s * sr)
    end_samp = int(end_s * sr)
    
    augmented = audio.copy()
    
    # Apply fades
    fade_samples = int(fade_ms * sr / 1000)
    
    if fade_samples > 0:
        # Fade out before zero region
        fade_start = max(0, start_samp - fade_samples)
        if fade_start < start_samp:
            fade_curve = np.linspace(1.0, 0.0, start_samp - fade_start)
            augmented[fade_start:start_samp] *= fade_curve
        
        # Fade in after zero region
        fade_end = min(len(audio), end_samp + fade_samples)
        if end_samp < fade_end:
            fade_curve = np.linspace(0.0, 1.0, fade_end - end_samp)
            augmented[end_samp:fade_end] *= fade_curve
    
    # Zero the segment
    augmented[start_samp:end_samp] = 0.0
    
    event = {
        "type": "missing",
        "label": 5,
        "start": start_s,
        "end": end_s
    }
    
    return augmented, event


# ============================================================================
# Frame Label Generation
# ============================================================================

def frames_from_time(time_s: float, hop_s: float) -> int:
    """Convert time in seconds to frame index."""
    return int(np.ceil(time_s / hop_s))


def generate_frame_labels(events: List[Dict], num_frames: int, 
                         hop_s: float) -> np.ndarray:
    """
    Generate frame-level labels from distortion events.
    
    Args:
        events: List of event dicts with keys: type, label, start, end
        num_frames: Total number of frames in utterance
        hop_s: Frame hop size in seconds
        
    Returns:
        Frame labels array of shape (num_frames,)
        
    Label priority (higher overwrites lower):
        5: missing (highest priority)
        4: packet_loss
        3: interference
        2: rir
        1: noise
        0: clean (lowest priority)
    """
    labels = np.zeros(num_frames, dtype=np.int64)
    
    # Sort events by priority (label value serves as priority)
    sorted_events = sorted(events, key=lambda x: x.get("label", 0))
    
    for event in sorted_events:
        start_frame = frames_from_time(event["start"], hop_s)
        end_frame = frames_from_time(event["end"], hop_s)
        
        # Clamp to valid range
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        
        if start_frame < end_frame:
            labels[start_frame:end_frame] = event["label"]
    
    return labels


# ============================================================================
# Word-level Augmentation
# ============================================================================

def apply_word_level_augmentations(
    audio: np.ndarray, sr: int,
    word_timestamps: List[Dict],
    noise_files: Optional[List[str]] = None,
    interference_files: Optional[List[str]] = None,
    p_noise: float = 0.3,
    p_interference: float = 0.2,
    p_packet_loss: float = 0.2,
    p_missing: float = 0.1
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Apply random augmentations at word level.
    
    Args:
        audio: Input audio
        sr: Sample rate
        word_timestamps: List of dicts with keys: word, start, end
        noise_files: Pool of noise files
        interference_files: Pool of interference speech files
        p_noise: Probability of applying noise to a word
        p_interference: Probability of applying interference to a word
        p_packet_loss: Probability of applying packet loss to a word
        p_missing: Probability of missing a word
        
    Returns:
        (augmented_audio, events_list)
    """
    augmented = audio.copy()
    events = []
    
    for word_info in word_timestamps:
        start_s = word_info["start"]
        end_s = word_info["end"]
        segment_range = (start_s, end_s)
        
        # Apply augmentations with probability
        rand = np.random.random()
        
        if rand < p_missing:
            # Missing segment (highest priority)
            augmented, event = zero_segment(augmented, sr, segment_range)
            events.append(event)
        
        elif rand < p_missing + p_packet_loss:
            # Packet loss
            bitrate = np.random.choice([6, 8, 12, 16])
            augmented, event = apply_packet_loss_augmentation(
                augmented, sr, segment_range, bitrate
            )
            events.append(event)
        
        elif rand < p_missing + p_packet_loss + p_interference:
            # Interference
            if interference_files:
                augmented, event = apply_interference_speaker_augmentation(
                    augmented, sr, interference_files,
                    apply_to_segment=True, segment_range=segment_range
                )
                events.append(event)
        
        elif rand < p_missing + p_packet_loss + p_interference + p_noise:
            # Noise
            if noise_files:
                augmented, event = apply_background_noise_augmentation(
                    augmented, sr, noise_files,
                    apply_to_segment=True, segment_range=segment_range
                )
                events.append(event)
    
    return augmented, events
