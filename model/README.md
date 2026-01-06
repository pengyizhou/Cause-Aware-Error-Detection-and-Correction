# Model Evaluation

This directory contains scripts and models for evaluating error detection and classification on dialogue/ASR data. The evaluation pipeline uses multiple trained models to predict different types of errors and then evaluates their performance at the word level.

## Overview

The evaluation system combines predictions from multiple models:
1. **Sound Event Detection** (6-class): Classifies distortion types (clean, noise, RIR, interference, packet loss, missing)
2. **Deletion Error Detection** (2-class): Detects deletion errors in ASR output
3. **Comprehension/Understanding Detection** (2-class): Predicts intelligibility (can understand vs cannot understand)
4. **Perception/Hearing Fault Detection** (2-class): Predicts hearing faults (can hear clearly vs cannot hear clearly)

These predictions are then combined and evaluated against ground truth word-level errors.

## Directory Structure

```
model/
├── README.md                          # This file
├── classifier_eval.sh                 # Main evaluation script (runs all models + evaluation)
├── evaluate_word_level_errors.py     # Word-level error evaluation script
├── Parakeet-encoder-linear/          # Encoder-based models (sound events, deletion)
│   ├── averaged_model_sound_event/   # Sound event detection checkpoint
│   └── averaged_model_deletion_only/ # Deletion detection checkpoint
└── Parakeet-joiner-linear/           # Joiner-based models (comprehension, perception)
    ├── averaged_model_comprehension/  # Comprehension/understanding checkpoint
    └── averaged_model_perception/     # Perception/hearing fault checkpoint
```

## Evaluation Workflow

```
Input Data (Parquet files)
    ↓
1. Run Sound Event Detection (Parakeet-encoder-linear)
    ↓
2. Run Deletion Detection (Parakeet-encoder-linear)
    ↓
3. Run Comprehension Detection (Parakeet-joiner-linear)
    ↓
4. Run Perception Detection (Parakeet-joiner-linear)
    ↓
5. Evaluate Word-Level Errors (evaluate_word_level_errors.py)
    ↓
Output: Marked text + Metrics
```

## Quick Start

### Complete Evaluation Pipeline

Run the complete evaluation pipeline using `classifier_eval.sh`:

```bash
# Update paths in classifier_eval.sh, then:
bash classifier_eval.sh
```

This script:
1. Runs inference with all 4 models
2. Combines predictions
3. Evaluates word-level errors
4. Outputs marked text and metrics

### Individual Model Evaluation

You can also run individual model evaluations:

```bash
# Sound Event Detection
python ./Parakeet-encoder-linear/decode.py \
    --data_dir <path/to/parquet> \
    --output_file <path/to/output.txt> \
    --checkpoint ./Parakeet-encoder-linear/averaged_model_sound_event \
    --projector_hidden_dim 640 \
    --use_encoder_projection \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --num_labels 6

# Deletion Detection
python ./Parakeet-encoder-linear/decode.py \
    --data_dir <path/to/parquet> \
    --output_file <path/to/output.txt> \
    --checkpoint ./Parakeet-encoder-linear/averaged_model_deletion_only \
    --projector_hidden_dim 640 \
    --use_encoder_projection \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --num_labels 2

# Comprehension Detection
python ./Parakeet-joiner-linear/decode.py \
    --data_dir <path/to/parquet> \
    --output_file <path/to/output.txt> \
    --checkpoint ./Parakeet-joiner-linear/averaged_model_comprehension \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --num_labels 2

# Perception Detection
python ./Parakeet-joiner-linear/decode.py \
    --data_dir <path/to/parquet> \
    --output_file <path/to/output.txt> \
    --checkpoint ./Parakeet-joiner-linear/averaged_model_perception \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --num_labels 2
```

## Scripts

### `classifier_eval.sh`

Main evaluation script that orchestrates the complete evaluation pipeline.

**What it does:**
1. Runs inference with all 4 models on dialogue datasets
2. Generates predictions for each model
3. Evaluates word-level errors by combining all predictions
4. Outputs marked text and metrics

**Configuration:**
- Update `datasets` array with dataset names
- Update `<PATH_TO_DATA>` with path to parquet data directories
- Update `<PATH_TO_OUTPUT>` with path for output files
- Update `<PATH_TO_ASR_RESULTS>` with path to ASR decoding results
- Update `<NODE_NAME>` for SLURM node assignment

**Output:**
- `result_encoder/event.txt`: Sound event predictions
- `result_encoder/deletion.txt`: Deletion predictions
- `result_joint/understanding.txt`: Comprehension predictions
- `result_joint/hearing_fault.txt`: Perception predictions
- `result_joint/understanding_marked.jsonl`: Marked text with errors
- `result_joint/understanding_metrics.json`: Evaluation metrics

**Usage:**
```bash
# Update paths in the script, then:
bash classifier_eval.sh
```

### `evaluate_word_level_errors.py`

Evaluates classifier predictions at the word level by:
1. Aligning reference and ASR transcriptions to find ground truth errors
2. Mapping frame-level predictions to words using token timestamps
3. Computing metrics (false-alarm rate, error recall, precision, F1)
4. Marking error words in the output text

**Required Arguments:**
- `--transcription`: Path to ground truth transcription file
- `--recog`: Path to ASR recognition output file
- `--token-timestamps`: Path to token timestamps file
- `--output`: Path to output file (marked text)

**Optional Arguments:**
- `--hearing-fault-predictions`: Path to hearing fault prediction JSONL file
- `--intelligibility-predictions`: Path to intelligibility prediction JSONL file
- `--deletion-predictions`: Path to deletion prediction JSONL file
- `--output-metrics`: Path to output metrics file (JSON format)
- `--frame-duration`: Frame duration in seconds (default: 0.08)

**Usage:**
```bash
python evaluate_word_level_errors.py \
    --transcription <path/to/transcription.txt> \
    --recog <path/to/recog.txt> \
    --token-timestamps <path/to/token_timestamps.txt> \
    --intelligibility-predictions <path/to/understanding.jsonl> \
    --hearing-fault-predictions <path/to/hearing_fault.jsonl> \
    --deletion-predictions <path/to/deletion.jsonl> \
    --output <path/to/output.jsonl> \
    --output-metrics <path/to/metrics.json>
```

**Input Format:**

**Transcription file** (`transcription.txt`):
```
uttid1 reference text here
uttid2 another reference text
```

**Recognition file** (`recog.txt`):
```
uttid1 ASR hypothesis text here
uttid2 another ASR hypothesis
```

**Token timestamps file** (`token_timestamps.txt`):
```
uttid1    [{'char': ['h', 'e', 'l', 'l', 'o'], 'start': 0.0, 'end': 0.5}, ...]
uttid2    [{'char': ['w', 'o', 'r', 'l', 'd'], 'start': 0.5, 'end': 1.0}, ...]
```

**Prediction JSONL file** (e.g., `understanding.jsonl`):
```jsonl
{"file_id": "uttid1", "predictions": [0, 0, 1, 1, 0, ...], "num_frames": 100}
{"file_id": "uttid2", "predictions": [0, 1, 1, 0, 0, ...], "num_frames": 150}
```

**Output Format:**

**Marked text** (`output.jsonl`):
```jsonl
{"file_id": "uttid1", "text": "hello <err>world</err> here", "error_words": [1]}
{"file_id": "uttid2", "text": "another text", "error_words": []}
```

**Metrics** (`metrics.json`):
```json
{
    "overall": {
        "false_alarm_rate": 0.05,
        "error_recall": 0.85,
        "precision": 0.78,
        "f1": 0.81,
        "true_positives": 120,
        "false_positives": 34,
        "false_negatives": 21,
        "true_negatives": 1825
    },
    "per_file": [...]
}
```

## Models

### Parakeet-Encoder-Linear Models

Located in `Parakeet-encoder-linear/`:

1. **Sound Event Detection** (`averaged_model_sound_event/`)
   - **Task**: 6-class distortion classification
   - **Labels**: 0=Clean, 1=Noise, 2=RIR, 3=Interference, 4=Packet Loss, 5=Missing
   - **Input**: Raw audio (processed through Parakeet encoder)
   - **Architecture**: Encoder + Projection (640-dim) + CNN classifier (5 layers, kernel=5)

2. **Deletion Detection** (`averaged_model_deletion_only/`)
   - **Task**: 2-class deletion error detection
   - **Labels**: 0=Clean, 1=Deletion error
   - **Input**: Raw audio (processed through Parakeet encoder)
   - **Architecture**: Encoder + Projection (640-dim) + CNN classifier (5 layers, kernel=5)

### Parakeet-Joiner-Linear Models

Located in `Parakeet-joiner-linear/`:

1. **Comprehension/Understanding** (`averaged_model_comprehension/`)
   - **Task**: 2-class intelligibility detection
   - **Labels**: 0=Intelligible, 1=Unintelligible
   - **Input**: Joint embeddings (640-dim) from Parakeet
   - **Architecture**: CNN classifier (5 layers, kernel=5)

2. **Perception/Hearing Fault** (`averaged_model_perception/`)
   - **Task**: 2-class hearing fault detection
   - **Labels**: 0=Can hear clearly, 1=Cannot hear clearly
   - **Input**: Joint embeddings (640-dim) from Parakeet
   - **Architecture**: CNN classifier (5 layers, kernel=5)

## Evaluation Metrics

The evaluation script computes the following metrics:

### Word-Level Metrics

- **False Alarm Rate (FAR)**: Proportion of correctly recognized words that are incorrectly flagged as errors
  - Formula: `FAR = FP / (FP + TN)`
  - Lower is better

- **Error Recall**: Proportion of actual errors that are correctly detected
  - Formula: `Recall = TP / (TP + FN)`
  - Higher is better

- **Precision**: Proportion of predicted errors that are actual errors
  - Formula: `Precision = TP / (TP + FP)`
  - Higher is better

- **F1 Score**: Harmonic mean of precision and recall
  - Formula: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
  - Higher is better

### Confusion Matrix Components

- **True Positives (TP)**: Errors correctly predicted as errors
- **False Positives (FP)**: Correct words incorrectly predicted as errors
- **False Negatives (FN)**: Errors missed by predictions
- **True Negatives (TN)**: Correct words correctly predicted as correct

## Prediction Combination Strategy

When multiple prediction sources are provided:

1. **Hearing Fault + Intelligibility**: Combined using max operation
   - If either model predicts error (label=1), the word is marked as error
   - This creates a union of error predictions

2. **Deletion Predictions**: Handled separately
   - Uses frame-level predictions to detect deletion errors
   - Groups consecutive deletion frames
   - Marks deletions based on absence of tokens in predicted frames

## Error Alignment

The evaluation script uses dynamic programming (Levenshtein distance) to align reference and hypothesis transcriptions:

- **Match**: Words that are identical (no error)
- **Substitution**: Words that differ (error)
- **Insertion**: Words in hypothesis but not in reference (error)
- **Deletion**: Words in reference but not in hypothesis (error)

Punctuation is removed for alignment but preserved in output.

## Token-to-Word Mapping

Frame-level predictions are mapped to words using token timestamps:

1. **Token Timestamps**: Each token has `start` and `end` times
2. **Frame-to-Token Mapping**: Frames are mapped to tokens based on time overlap
3. **Token-to-Word Mapping**: Tokens are grouped into words
4. **Word Prediction**: Word is marked as error if any of its tokens are predicted as error

## Dependencies

- Python 3.7+
- PyTorch >= 1.10.0
- NumPy
- Standard library (json, argparse, re, collections)

## Configuration

Before running evaluation, update the following in `classifier_eval.sh`:

- `<DATASET1>`, `<DATASET2>`, etc.: Dataset names
- `<PATH_TO_DATA>`: Path to parquet data directories
- `<PATH_TO_OUTPUT>`: Path for output files
- `<PATH_TO_ASR_RESULTS>`: Path to ASR decoding results
- `<NODE_NAME>`: SLURM node name (if using SLURM)

## Notes

- **Model Checkpoints**: All models use averaged checkpoints for better stability
- **Frame Duration**: Default frame duration is 80ms (0.08 seconds)
- **Deletion Detection**: Uses frame-level predictions with tolerance for token alignment
- **Prediction Combination**: Hearing fault and intelligibility predictions are combined using max operation
- **Output Format**: Marked text uses `<err>...</err>` tags to indicate error words
- **Metrics**: Both per-file and overall metrics are computed and saved

## Example Workflow

```bash
# 1. Prepare data in parquet format (see Parakeet-encoder-linear/README.md and Parakeet-joiner-linear/README.md)

# 2. Run complete evaluation
bash classifier_eval.sh

# 3. Check results
cat <PATH_TO_OUTPUT>/<dataset>/result_joint/understanding_metrics.json
cat <PATH_TO_OUTPUT>/<dataset>/result_joint/understanding_marked.jsonl
```

## Citation

If you use these evaluation tools, please cite:
- Parakeet TDT model (NVIDIA)
- NeMo toolkit
- Relevant datasets used for evaluation

