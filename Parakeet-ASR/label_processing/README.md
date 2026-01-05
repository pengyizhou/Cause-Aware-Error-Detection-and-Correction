# Label Processing Pipeline

This directory contains scripts for processing ASR results and generating frame-level and token-level labels for different tasks: comprehension, perception, and distortion event detection.

## Overview

The label processing pipeline takes ASR decoding results (transcriptions, timestamps, token timestamps) and generates various types of labels for downstream tasks. The pipeline consists of 5 sequential steps that must be executed in order.

## Pipeline Workflow

```
ASR Decoding Results
    ↓
1. process_results.sh (Filter corrupted timestamps)
    ↓
2. process_timestamps_distorted.sh (Find overlapping utterances)
    ↓
3. process_understand.sh (Generate comprehension labels)
    ↓
4. process_token_labels_deletion2.sh (Generate perception labels)
    ↓
5. process_token_labels_for_sound_events.sh (Generate distortion event labels)
```

## Directory Structure

```
label_processing/
├── README.md                          # This file
├── process_results.sh                 # Step 1: Filter corrupted timestamps
├── process_timestamps_distorted.sh    # Step 2: Find overlapping utterances
├── process_understand.sh              # Step 3: Generate comprehension labels
├── process_token_labels_deletion2.sh  # Step 4: Generate perception labels
├── process_token_labels_for_sound_events.sh  # Step 5: Generate distortion event labels
├── cleanup_token_timestamps.py       # Utility: Remove overlapping timestamps
├── compare_timestamps_CSID.py        # Utility: Compare clean vs distorted timestamps
├── generate_token_labels.py          # Utility: Generate token-level labels
└── generate_labels_with_npy_labels.py # Utility: Generate labels from .npy files
```

## Step-by-Step Pipeline

### Step 1: Filter Corrupted Timestamps (`process_results.sh`)

**Purpose**: Remove utterances with corrupted or empty timestamps from ASR decoding results.

**What it does**:
- Filters out utterances with empty token timestamps (`[]`)
- Cleans up token timestamps to remove overlapping segments
- Outputs cleaned timestamp files with `.nooverlap` extension

**Input**:
- `token_timestamps.txt` files from ASR decoding results

**Output**:
- `token_timestamps.txt.clean`: Filtered timestamps (no empty entries)
- `token_timestamps.txt.nooverlap`: Cleaned timestamps without overlaps

**Usage**:
```bash
# Update datasets and paths in the script
bash process_results.sh
```

**Configuration**:
- Update `datasets` variable with dataset names
- Update `parts` variable with partition/distortion types
- Update `<PATH_TO_RESULTS>` with your results directory

### Step 2: Find Overlapping Utterances (`process_timestamps_distorted.sh`)

**Purpose**: For distortion data, find utterances that exist in both clean and distorted versions for fair comparison.

**What it does**:
- Sorts clean and distorted timestamp files by utterance ID
- Finds common utterances between clean and distorted data
- Creates filtered files containing only overlapping utterances:
  - `.clean.txt`: Clean timestamps for overlapping utterances
  - `.final.txt`: Distorted timestamps for overlapping utterances

**Input**:
- `token_timestamps.txt.nooverlap` from Step 1 (both clean and distorted)

**Output**:
- `token_timestamps.txt.nooverlap.clean.txt`: Clean timestamps (overlapping only)
- `token_timestamps.txt.nooverlap.final.txt`: Distorted timestamps (overlapping only)

**Usage**:
```bash
# Update datasets and paths in the script
bash process_timestamps_distorted.sh
```

**Configuration**:
- Update `datasets` variable with dataset names
- Update `parts` variable with distortion types
- Update `<PATH_TO_RESULTS>` with your results directory

**Note**: This step is only needed for distorted data. Clean data can skip to Step 3.

### Step 3: Generate Comprehension Labels (`process_understand.sh`)

**Purpose**: Generate token-level labels for comprehension/intelligibility tasks.

**What it does**:
- Compares ASR transcriptions with reference transcriptions
- Generates labels indicating correct/incorrect tokens
- Uses `--deletion_strategy skip` to handle deletion errors

**Input**:
- `timestamps.txt`: Word-level timestamps
- `token_timestamps.txt`: Token-level timestamps
- `transcription.txt`: Reference transcriptions

**Output**:
- `token_timestamps.labels.txt`: Token-level comprehension labels

**Usage**:
```bash
# Update datasets and paths in the script
bash process_understand.sh
```

**Configuration**:
- Update `datasets` variable with dataset names
- Update `parts` variable (typically `"all_clean"` for clean data)
- Update `<PATH_TO_RESULTS>` with your results directory

**Label Format**:
- Format: `uttid\t[frame:label,frame:label,...]`
- Labels indicate correct/incorrect tokens for comprehension evaluation

### Step 4: Generate Perception Labels (`process_token_labels_deletion2.sh`)

**Purpose**: Generate perception labels with special handling for deletion errors (labeled as 2).

**What it does**:
- Compares clean (reference) and distorted (hypothesis) token timestamps
- Identifies Substitution, Insertion, and Deletion errors
- **Distinguishes deletion errors with label 2** (other errors use label 1)
- Uses dynamic programming alignment (Levenshtein distance)

**Input**:
- `token_timestamps.txt.nooverlap.clean.txt`: Clean reference timestamps (from Step 2)
- `token_timestamps.txt.nooverlap.final.txt`: Distorted hypothesis timestamps (from Step 2)

**Output**:
- `token_timestamps.txt.nooverlap.labels.deletion2.norm.txt`: Perception labels with deletion distinction

**Usage**:
```bash
# Update datasets and paths in the script
bash process_token_labels_deletion2.sh
```

**Configuration**:
- Update `datasets` variable with dataset names
- Update `parts` variable with distortion types
- Update `<PATH_TO_RESULTS>` with your results directory

**Label Format**:
- Format: `uttid\t[start_offset:label,start_offset:label,...]`
- Label 0: Correct token
- Label 1: Substitution/Insertion error
- Label 2: **Deletion error** (distinguished from other errors)

**Note**: Requires Step 2 to be completed first (needs `.clean.txt` and `.final.txt` files).

### Step 5: Generate Distortion Event Labels (`process_token_labels_for_sound_events.sh`)

**Purpose**: Generate distortion event labels from frame-level .npy label files.

**What it does**:
- Loads frame-level labels from `.npy` files (20ms resolution)
- Downsamples to token-level labels (80ms resolution)
- Converts frame labels to token timestamp format

**Input**:
- Directory containing `.npy` label files (one per utterance)
- Each `.npy` file contains frame-level distortion labels (0-5)

**Output**:
- `token_timestamps.txt.nooverlap.labels.sound_events.txt`: Token-level distortion event labels

**Usage**:
```bash
# Update datasets and paths in the script
bash process_token_labels_for_sound_events.sh
```

**Configuration**:
- Update `datasets` variable with dataset names
- Update `parts` variable with distortion types
- Update `<PATH_TO_LABEL_DATA>` with path to .npy label files
- Update `<PATH_TO_RESULTS>` with your results directory

**Label Format**:
- Format: `uttid\t[frame:label,frame:label,...]`
- Frame indices are at 80ms resolution (downsampled from 20ms)
- Labels: 0-5 (distortion types: clean, noise, RIR, interference, packet_loss, missing)

**Frame Resolution**:
- Input: 20ms per frame (from .npy files)
- Output: 80ms per frame (4:1 downsampling ratio)

## Utility Scripts

### `cleanup_token_timestamps.py`

Removes overlapping token timestamps from timestamp files.

**Usage**:
```bash
python cleanup_token_timestamps.py -i <input_file> -o <output_file>
```

### `compare_timestamps_CSID.py`

Compares clean and distorted token timestamps to identify errors (Correct, Substitution, Insertion, Deletion).

**Usage**:
```bash
python compare_timestamps_CSID.py <clean_file> <final_file> <output_file> [--full_range]
```

**Options**:
- `--full_range`: Expand labels to all frames from start_offset to end_offset

### `generate_token_labels.py`

Generates token-level labels from timestamps and transcriptions.

**Usage**:
```bash
python generate_token_labels.py \
    --timestamps <timestamps.txt> \
    --token_timestamps <token_timestamps.txt> \
    --transcription <transcription.txt> \
    --output <output.txt> \
    --deletion_strategy <strategy>
```

### `generate_labels_with_npy_labels.py`

Generates token-level labels from frame-level .npy label files.

**Usage**:
```bash
python generate_labels_with_npy_labels.py <label_dir> <output_file> [--method <method>]
```

**Options**:
- `--method`: Downsampling method (`first`, `max`, or `majority`)

## Complete Pipeline Example

```bash
# Step 1: Filter corrupted timestamps
bash process_results.sh

# Step 2: Find overlapping utterances (for distorted data)
bash process_timestamps_distorted.sh

# Step 3: Generate comprehension labels (for clean data)
bash process_understand.sh

# Step 4: Generate perception labels (for distorted data)
bash process_token_labels_deletion2.sh

# Step 5: Generate distortion event labels (for distorted data)
bash process_token_labels_for_sound_events.sh
```

## Label Types

### Comprehension Labels (Step 3)
- **Purpose**: Intelligibility/comprehension evaluation
- **Format**: Token-level correct/incorrect labels
- **Use Case**: Understanding task evaluation

### Perception Labels (Step 4)
- **Purpose**: ASR error detection with deletion distinction
- **Format**: Token-level error labels
- **Label Values**:
  - 0: Correct
  - 1: Substitution/Insertion error
  - 2: **Deletion error** (special label)
- **Use Case**: Perception task evaluation, deletion error detection

### Distortion Event Labels (Step 5)
- **Purpose**: Frame-level distortion type classification
- **Format**: Frame-level labels (downsampled to token-level)
- **Label Values**:
  - 0: Clean
  - 1: Noise
  - 2: RIR
  - 3: Interference
  - 4: Packet loss
  - 5: Missing
- **Use Case**: Distortion event detection and classification

## Dependencies

- Python 3.7+
- NumPy
- Standard Unix utilities (grep, sort, cut)

## Notes

- **Execution Order**: Steps must be executed sequentially (1 → 2 → 3/4/5)
- **Step 2 Prerequisite**: Step 4 requires Step 2 output (`.clean.txt` and `.final.txt` files)
- **Clean vs Distorted**: 
  - Clean data: Steps 1 → 3
  - Distorted data: Steps 1 → 2 → 4/5
- **Frame Resolution**: Step 5 downsamples from 20ms to 80ms frames
- **Deletion Distinction**: Step 4 uses label 2 specifically for deletion errors to distinguish them from other error types

## Configuration

All scripts use placeholder variables that must be updated:
- `<DATASET1>`, `<DATASET2>`, etc.: Dataset names
- `<PATH_TO_RESULTS>`: Path to ASR decoding results directory
- `<PATH_TO_LABEL_DATA>`: Path to .npy label files directory
- `<JOB_ID>`: SLURM job dependency ID (if using SLURM)

Update these placeholders in each script before execution.

