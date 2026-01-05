# Data Preparation: Speech Distortion Simulation

This directory contains scripts for simulating various types of speech distortions on clean audio corpora. The tools generate distorted audio files along with frame-level labels (at 50 Hz, 20ms frame hop) indicating the type of distortion present in each frame.

## Overview

The data preparation pipeline takes clean speech audio and applies realistic distortions to create training and evaluation datasets for distortion detection and correction models. Each script processes Kaldi-style `wav.scp` and `segments` files and outputs:

- **Distorted audio files** (`.wav`)
- **Frame-level labels** (`.npy`) - numpy arrays with distortion type labels
- **Metadata** (`.json`) - information about applied distortions

## Distortion Types

The system supports the following distortion types (with corresponding label values):

- **0: Clean** - No distortion
- **1: Noise** - Background noise (additive)
- **2: RIR** - Room Impulse Response (reverberation)
- **3: Interference** - Interfering speakers
- **4: Packet Loss** - Network packet loss (zeroed segments)
- **5: Missing** - Totally missing segments

## Directory Structure

```
data_prepare/
├── README.md                          # This file
├── run.sh                             # Main orchestration script
├── distortion_utils.py                # Core distortion simulation utilities
├── distortion_augment.py              # Advanced augmentation functions
├── generate_clean.py                  # Generate clean audio with zero labels
├── generate_noise.py                  # Add background noise
├── generate_rir.py                    # Apply room impulse response
├── generate_interference.py           # Add interfering speakers
├── generate_packet_loss.py            # Simulate packet loss
├── generate_missing.py                # Simulate missing segments
├── generate_partial_noise.py          # Add noise to partial segments
├── generate_rir_noise.py              # Apply RIR + noise (combined)
├── generate_multi_distortion_with_rir.py    # Multiple distortions + RIR
├── generate_multi_distortion_no_rir.py      # Multiple distortions (no RIR)
└── generate_all_distortions.py       # Apply all distortion types
```

## Core Components

### `distortion_augment.py`
Core augmentation library providing functions for:
- RIR application (`apply_rir`, `apply_rir_augmentation`)
- Background noise mixing (`mix_background_noise`, `apply_background_noise_augmentation`)
- Interference speaker addition (`apply_interference_speaker_augmentation`)
- Packet loss simulation (`apply_packet_loss_augmentation`)
- Missing segment simulation (`zero_segment`)
- Frame label generation (`generate_frame_labels`)
- Word-level augmentations (`apply_word_level_augmentations`)

### `distortion_utils.py`
Utility classes and functions:
- `DistortionSimulator` class for basic distortion simulation
- `apply_distortion_with_labels` for integrated distortion application

## Usage

### Basic Workflow

1. **Prepare input data** in Kaldi format:
   - `wav.scp`: Maps recording IDs to audio file paths (supports sox pipe commands)
   - `segments`: Maps segment IDs to recording IDs and time boundaries

2. **Run generation scripts** for desired distortion types

3. **Output format**: Each segment produces:
   - `{segment_id}.wav` - Distorted audio (16kHz, mono)
   - `{segment_id}.npy` - Frame-level labels (50 Hz, 20ms hop)
   - `{segment_id}.json` - Metadata with distortion parameters

### Example: Generate Clean Data

```bash
python generate_clean.py \
    --wav_scp data/clean/wav.scp \
    --segments data/clean/segments \
    --output_dir output/clean \
    --sample_rate 16000
```

### Example: Generate Noisy Data

```bash
python generate_noise.py \
    --wav_scp data/clean/wav.scp \
    --segments data/clean/segments \
    --noise_wav_scp data/noise/wav.scp \
    --output_dir output/noise \
    --snr_range -5 20
```

### Example: Generate RIR + Noise

```bash
python generate_rir_noise.py \
    --wav_scp data/clean/wav.scp \
    --segments data/clean/segments \
    --rir_wav_scp data/rir/wav.scp \
    --noise_wav_scp data/noise/wav.scp \
    --output_dir output/rir_noise \
    --snr_range -5 20 \
    --snr_threshold 10
```

### Example: Generate Multi-Distortion Data

```bash
python generate_multi_distortion_with_rir.py \
    --wav_scp data/clean/wav.scp \
    --segments data/clean/segments \
    --rir_wav_scp data/rir/wav.scp \
    --noise_wav_scp data/noise/wav.scp \
    --interference_wav_scp data/interference/wav.scp \
    --output_dir output/multi_dist_rir
```

## Script Descriptions

### Single Distortion Scripts

- **`generate_clean.py`**: Creates clean audio copies with all-zero labels
- **`generate_noise.py`**: Adds background noise with configurable SNR range
- **`generate_rir.py`**: Applies room impulse response for reverberation
- **`generate_interference.py`**: Adds interfering speakers with configurable SIR
- **`generate_packet_loss.py`**: Simulates network packet loss with configurable bitrates
- **`generate_missing.py`**: Simulates totally missing segments

### Combined Distortion Scripts

- **`generate_rir_noise.py`**: Applies RIR first, then adds noise. Labels based on SNR threshold:
  - SNR < threshold → Label 1 (noise)
  - SNR ≥ threshold → Label 2 (RIR)
- **`generate_partial_noise.py`**: Noise applied only to partial segments

### Multi-Distortion Scripts

- **`generate_multi_distortion_with_rir.py`**: 
  - RIR applied to entire segment
  - 2-4 distortion types (packet_loss, missing, noise, interference)
  - Each type occurs 1-2 times
  - Distortions do not overlap
  
- **`generate_multi_distortion_no_rir.py`**: 
  - Same as above but without RIR

## Batch Processing with `run.sh`

The `run.sh` script orchestrates batch processing across multiple datasets and distortion types:

```bash
# Configure datasets and partitions
datalist="libri-train libri-valid spgi-test spgi-train spgi-valid"
partitions="rir_noise rir_noise_partial multi_dist_rir multi_dist_no_rir"

# Run the script
bash run.sh
```

The script:
1. Splits datasets into partitions (if needed)
2. Generates distorted data for each dataset-partition combination
3. Supports parallel execution across multiple nodes via SLURM

## Output Format

### Audio Files (`.wav`)
- Format: WAV, 16-bit PCM
- Sample rate: 16 kHz (configurable)
- Channels: Mono

### Label Files (`.npy`)
- Format: NumPy array, dtype `int64`
- Shape: `(num_frames,)`
- Frame rate: 50 Hz (20ms hop, configurable)
- Values: 0-5 (distortion type labels)

### Metadata Files (`.json`)
Example structure:
```json
{
  "segment_id": "utt_001",
  "recording_id": "rec_001",
  "start": 0.5,
  "end": 3.2,
  "distortion_type": "rir_noise_noisy",
  "label": 1,
  "snr_db": 8.5,
  "snr_threshold": 10.0,
  "duration_seconds": 2.7,
  "num_frames": 135
}
```

## Parameters

### Common Parameters

- `--wav_scp`: Path to Kaldi wav.scp file (required)
- `--segments`: Path to Kaldi segments file (required)
- `--output_dir`: Output directory for generated data (required)
- `--sample_rate`: Target sample rate in Hz (default: 16000)
- `--frame_hop_ms`: Frame hop in milliseconds (default: 20.0 for 50fps)
- `--max_segments`: Maximum segments to process (for testing)

### Distortion-Specific Parameters

**Noise:**
- `--noise_wav_scp`: Path to noise wav.scp file
- `--snr_range`: SNR range in dB (e.g., `-5 20`)

**RIR:**
- `--rir_wav_scp`: Path to RIR wav.scp file
- `--rir_segments`: Optional RIR segments file

**Interference:**
- `--interference_wav_scp`: Path to interference wav.scp file
- `--sir_range`: Signal-to-interference ratio range in dB

**Packet Loss:**
- `--bitrates`: List of bitrates to simulate (e.g., `1 2 4 8`)
- `--num_distortions_range`: Number of distortion events (e.g., `1 3`)

**Missing Segments:**
- `--segment_duration_range`: Duration range for missing segments in seconds

**RIR + Noise:**
- `--snr_threshold`: SNR threshold for label assignment (default: 10.0 dB)

## Dependencies

- Python 3.7+
- NumPy
- SciPy
- SoundFile
- PyTorch / TorchAudio (for some operations)
- tqdm (for progress bars)

## Notes

- All scripts support Kaldi-style wav.scp files, including sox pipe commands (ending with `|`)
- Audio is automatically resampled to target sample rate if needed
- Stereo audio is converted to mono
- Frame labels are generated at 50 Hz (20ms hop) by default
- Distortions are applied randomly within specified parameter ranges
- The system supports both word-level and frame-level distortion application

## Workflow Example

1. **Prepare clean corpus**:
   ```bash
   python generate_clean.py --wav_scp clean/wav.scp --segments clean/segments \
                            --output_dir data/clean
   ```

2. **Generate single distortion types**:
   ```bash
   python generate_noise.py --wav_scp clean/wav.scp --segments clean/segments \
                            --noise_wav_scp noise/wav.scp --output_dir data/noise \
                            --snr_range -5 20
   ```

3. **Generate combined distortions**:
   ```bash
   python generate_rir_noise.py --wav_scp clean/wav.scp --segments clean/segments \
                                 --rir_wav_scp rir/wav.scp --noise_wav_scp noise/wav.scp \
                                 --output_dir data/rir_noise --snr_range -5 20
   ```

4. **Generate multi-distortion data**:
   ```bash
   python generate_multi_distortion_with_rir.py \
       --wav_scp clean/wav.scp --segments clean/segments \
       --rir_wav_scp rir/wav.scp --noise_wav_scp noise/wav.scp \
       --interference_wav_scp interference/wav.scp \
       --output_dir data/multi_dist_rir
   ```

## Citation

If you use this data preparation pipeline, please cite the relevant papers and acknowledge the use of these tools.

### Datasets

**LibriSpeech:**
```
@inproceedings{panayotov2015librispeech,
  title={Librispeech: an ASR corpus based on public domain audio books},
  author={Panayotov, Vassil and Chen, Guoguo and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2015 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={5206--5210},
  year={2015},
  organization={IEEE}
}
```

**SPGISpeech 2.0:**
```
@article{grossman2025spgispeech,
  title={SPGISpeech 2.0: Transcribed multi-speaker financial audio for speaker-tagged transcription},
  author={Grossman, Raymond and Park, Taejin and Dhawan, Kunal and Titus, Andrew and Zhi, Sophia and Shchadilova, Yulia and Wang, Weiqing and Balam, Jagadeesh and Ginsburg, Boris},
  journal={arXiv preprint arXiv:2508.05554},
  year={2025}
}
```

**AESRC 2020:**
```
@inproceedings{shi2021accented,
  title={The Accented English Speech Recognition Challenge 2020: Open Datasets, Tracks, Baselines, Results and Methods},
  author={Shi, Xian and Yu, Fan and Lu, Yizhou and Liang, Yuhao and Feng, Qiangze and Wang, Daliang and Qian, Yanmin and Xie, Lei},
  booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6918--6922},
  year={2021},
  organization={IEEE}
}
```

### Resources

**MUSAN Corpus (for noise and interference):**
```
@misc{snyder2015musan,
  title={MUSAN: A Music, Speech, and Noise Corpus},
  author={Snyder, David and Chen, Guoguo and Povey, Daniel},
  year={2015},
  eprint={1510.08484},
  archivePrefix={arXiv},
  primaryClass={cs.SD}
}
```

**Opus Audio Codec (for packet loss simulation):**
```
@inproceedings{valin2012opus,
  title={Definition of the Opus Audio Codec},
  author={Valin, Jean-Marc and Vos, Koen and Terriberry, Timothy B},
  booktitle={Request for Comments: 6716},
  year={2012},
  publisher={IETF}
}
```
