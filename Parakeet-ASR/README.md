# Parakeet-ASR: Speech Recognition and Error Detection

This directory contains scripts for running ASR inference using NVIDIA's Parakeet TDT model and evaluating error detection methods, including entropy-based confidence scoring.

## Overview

The scripts in this directory provide:
- **ASR Inference**: Transcribe audio using Parakeet TDT 0.6B v2 model
- **Feature Extraction**: Extract joint embeddings (640-dim) and logits for downstream tasks
- **Confidence Scoring**: Compute entropy-based confidence scores for error detection
- **Error Evaluation**: Evaluate recall and precision of error detection methods

## Directory Structure

```
Parakeet-ASR/
├── README.md                          # This file
├── inference_with_joint_emb_logits.py # Main inference script (embeddings + logits + confidence)
├── evaluate_confidence_recall.py      # Evaluate confidence-based error detection
├── compute-wer.py                     # Word Error Rate computation tool
├── decode_accent.sh                   # Decode accented speech datasets with WER
└── decode_confidence.sh               # Decode with entropy-based confidence evaluation
```

## Core Components

### `inference_with_joint_emb_logits.py`

Main inference script that:
- Loads Parakeet TDT 0.6B v2 model
- Transcribes audio files
- Extracts joint embeddings (640-dim) for each frame
- Extracts logits (1030-dim) for each frame
- Computes entropy-based confidence scores
- Outputs transcriptions, timestamps, and confidence scores

**Output Files:**
- `recog.txt`: Transcription hypotheses (format: `uttid transcription`)
- `transcription.txt`: Reference transcriptions (if provided)
- `joint_embeddings.pt`: Dictionary mapping `uttid → Tensor[time, 640]`
- `logits.pt`: Dictionary mapping `uttid → Tensor[time, 1030]`
- `confidence.json`: Word-level confidence scores
- `timestamps.txt`: Word-level timestamps
- `token_timestamps.txt`: Token-level timestamps

**Usage:**
```bash
# From audio directory
python inference_with_joint_emb_logits.py \
    --audio_dir <path/to/audio/directory> \
    --result_dir <path/to/output>

# From wav.scp file
python inference_with_joint_emb_logits.py \
    --audio_scp <path/to/wav.scp> \
    --result_dir <path/to/output>
```

### `evaluate_confidence_recall.py`

Evaluates entropy-based confidence method for error detection:
- Converts confidence scores to error probabilities (1 - confidence)
- Finds threshold that achieves target False Positive Rate (FPR)
- Calculates recall for misrecognized words at that threshold
- Supports both pre-computed labels and hyp/ref alignment

**Usage:**
```bash
# Using hyp/ref alignment (recommended)
python evaluate_confidence_recall.py \
    --confidence_json <path/to/confidence.json> \
    --hyp_file <path/to/hyp.txt> \
    --ref_file <path/to/ref.txt> \
    --target_fpr <target_fpr_value>

# Using pre-computed labels
python evaluate_confidence_recall.py \
    --confidence_json <path/to/confidence.json> \
    --labels_file <path/to/labels.txt> \
    --target_fpr <target_fpr_value>
```

## Decode Scripts

### `decode_accent.sh`

Decodes accented speech datasets (e.g., AESRC 2020):
- Processes multiple partitions (train/valid/test)
- Extracts transcriptions, joint embeddings, and logits
- Computes Word Error Rate (WER) using `compute-wer.py`
- Sorts transcriptions and references for WER computation

**Configuration:**
- Update `src` variable with dataset path
- Update `parts` variable with partitions to process (default: "valid test train")
- Update `<DATASET_NAME>` in result paths

**Output:**
- `recog.txt`: ASR hypotheses
- `transcription.txt`: Reference transcriptions (sorted)
- `wer_result.txt`: WER computation results
- `joint_embeddings.pt`: Joint embeddings for downstream tasks
- `logits.pt`: Logits for analysis
- `confidence.json`: Word-level confidence scores

**Usage:**
```bash
# Update paths in the script, then:
bash decode_accent.sh
```

### `decode_confidence.sh`

**Entropy-based confidence method for error detection:**
- Processes dataset partitions (e.g., test set)
- Extracts transcriptions and confidence scores
- Evaluates confidence-based error detection at target FPR
- Uses hyp/ref alignment to compute error labels

**Configuration:**
- Update `parts` variable with partitions to process (default: "test")
- Update `src` variable with dataset path
- Update `<DATASET_NAME>` in result paths
- Update `target_fpr` for confidence threshold (default: 0.0113)

**Output:**
- `recog.txt`: ASR hypotheses
- `transcription.txt`: Reference transcriptions
- `confidence.json`: Word-level confidence scores
- Evaluation metrics printed to console

**Usage:**
```bash
# Update paths and target_fpr in the script, then:
bash decode_confidence.sh
```

### `compute-wer.py`

Utility script for computing Word Error Rate (WER):
- Supports character-level and word-level WER
- Handles text normalization
- Outputs detailed WER statistics

**Usage:**
```bash
./compute-wer.py --char=1 --v=1 \
    <reference_file> <hypothesis_file> > <output_file>
```

## Entropy-Based Confidence Method

The confidence scoring uses **entropy-based method** with the following configuration:

```python
ConfidenceConfig(
    preserve_word_confidence=True,
    aggregation="prod",  # Product aggregation
    exclude_blank=False,
    method_cfg=ConfidenceMethodConfig(
        name="entropy",
        entropy_type="tsallis",
        alpha=0.33,  # Tsallis entropy parameter
        entropy_norm="exp",  # Exponential normalization
    )
)
```

**How it works:**
1. Computes frame-level entropy from logits
2. Aggregates to token-level and word-level confidence
3. Converts to error probability: `error_prob = 1 - confidence`
4. Thresholds at target FPR to detect errors

**Advantages:**
- No training required (unsupervised)
- Fast inference
- Works across different distortion types
- Calibrated per distortion type

## Output Format

### Transcription Files

**`recog.txt`** (hypotheses):
```
uttid1 transcription text here
uttid2 another transcription
```

**`transcription.txt`** (references):
```
uttid1 reference text here
uttid2 another reference
```

### Confidence JSON

**`confidence.json`**:
```json
{
    "uttid1": {
        "word_confidence": [0.95, 0.87, 0.92, ...]
    },
    "uttid2": {
        "word_confidence": [0.88, 0.91, 0.76, ...]
    }
}
```

### Embeddings and Logits

**`joint_embeddings.pt`**: PyTorch dictionary
```python
{
    "uttid1": Tensor[time, 640],
    "uttid2": Tensor[time, 640],
    ...
}
```

**`logits.pt`**: PyTorch dictionary
```python
{
    "uttid1": Tensor[time, 1030],
    "uttid2": Tensor[time, 1030],
    ...
}
```

## Usage Examples

### Example 1: Decode Accented Speech with WER

```bash
# Update paths in decode_accent.sh
bash decode_accent.sh
```

This will:
1. Run ASR inference on each partition
2. Extract joint embeddings and logits
3. Compute WER for each partition
4. Save results to `Parakeet-ASR/results/<DATASET_NAME>/<partition>/`

### Example 2: Evaluate Confidence-Based Error Detection

```bash
# Update paths and target_fpr in decode_confidence.sh
bash decode_confidence.sh
```

This will:
1. Run ASR inference on specified partitions
2. Extract confidence scores
3. Evaluate error detection at target FPR
4. Print recall and precision metrics

### Example 3: Extract Joint Embeddings for Training

```bash
python inference_with_joint_emb_logits.py \
    --audio_scp data/train/wav.scp \
    --result_dir results/train

# Use joint_embeddings.pt for training Parakeet-joiner-linear model
```

### Example 4: Evaluate Confidence Manually

```bash
python evaluate_confidence_recall.py \
    --confidence_json results/test/confidence.json \
    --hyp_file results/test/recog.txt \
    --ref_file results/test/transcription.txt \
    --target_fpr 0.05
```

### Example 5: Compute WER Manually

```bash
./compute-wer.py --char=1 --v=1 \
    results/test/transcription.txt \
    results/test/recog.txt \
    > results/test/wer_result.txt
```

## SLURM Configuration

All decode scripts are configured for SLURM job submission:
- Single GPU (`--gres=gpu:1`)
- 4 CPUs (`--cpus-per-task=4`)
- Node assignment via `-w <NODE_NAME>`

Update the `<NODE_NAME>` placeholder in each script before submission.

## Dependencies

- Python 3.7+
- PyTorch >= 1.10.0
- NeMo toolkit (for Parakeet models)
- NumPy
- scikit-learn
- CUDA-capable GPU

### Installing NeMo

The NeMo toolkit is required for loading and using Parakeet models. For detailed installation instructions, please refer to the [official NeMo installation guide](https://github.com/NVIDIA-NeMo/NeMo/?tab=readme-ov-file#conda--pip).

**Quick installation (Conda + Pip):**

```bash
# Create a fresh Conda environment
conda create --name nemo python==3.10.12
conda activate nemo

# Install NeMo with ASR support
pip install "nemo_toolkit[all]"
# Or for ASR only:
pip install "nemo_toolkit[asr]"
```

**Alternative: Install from source**

```bash
git clone https://github.com/NVIDIA/NeMo
cd NeMo
git checkout main  # or specific version/tag
pip install ".[all]"
```

For more installation options (NGC containers, domain-specific installations, etc.), see the [NeMo installation documentation](https://github.com/NVIDIA-NeMo/NeMo/?tab=readme-ov-file#conda--pip).

## Notes

- **Model**: Uses `nvidia/parakeet-tdt-0.6b-v2` (600M parameters)
- **Batch Size**: Inference uses `batch_size=1` for sequential processing
- **Confidence Method**: Entropy-based (Tsallis entropy with α=0.33)
- **Frame Rate**: Joint embeddings at 50 Hz (20ms frame hop)
- **Output Format**: All outputs are saved to `result_dir` specified in scripts
- **WER Computation**: `decode_accent.sh` automatically computes WER using `compute-wer.py`
- **FPR Calibration**: Target FPR can be adjusted in `decode_confidence.sh` based on desired precision/recall trade-off

## Citation

If you use this code, please cite:
- Parakeet TDT model (NVIDIA)
- NeMo toolkit
- Relevant datasets used for evaluation

