# Parakeet-Encoder-Linear: Frame-Level Distortion Detection

This directory contains a PyTorch implementation for frame-level speech distortion detection using NVIDIA's Parakeet TDT encoder with a linear/CNN classification head.

## Overview

The model uses a pre-trained Parakeet encoder (frozen) to extract frame-level features from audio, then applies a trainable classification head to predict distortion types at 50 Hz (20ms frame hop). The model supports both binary classification (e.g., hearing fault detection) and multi-class classification (e.g., 6-class distortion types).

## Model Architecture

### Base Encoder
- **Parakeet TDT 0.6B v2** (`nvidia/parakeet-tdt-0.6b-v2`)
- Encoder hidden dimension: 1024
- Preprocessor and encoder are frozen by default

### Classification Head Options

1. **Linear Classifier** (default)
   - Single linear layer: `encoder_dim → num_labels`

2. **Two-Layer Projector + Linear**
   - Optional hidden projection layer: `encoder_dim → projector_hidden_dim → num_labels`
   - Set `--projector_hidden_dim` to enable

3. **Encoder Projection + Classifier**
   - Uses Parakeet's joint network encoder projection (1024 → 640)
   - Then applies classifier: `640 → num_labels`
   - Set `--use_encoder_projection` to enable

4. **CNN Classifier**
   - Convolutional layers for local window processing
   - Configurable kernel size and number of layers
   - Set `--cnn_kernel_size` and `--cnn_num_layers` to enable

5. **Conv-Transformer**
   - 2 CNN layers + 3 Transformer encoder layers
   - Set `--use_conv_transformer` to enable

## Directory Structure

```
Parakeet-encoder-linear/
├── README.md                 # This file
├── model.py                  # Model definition (ParakeetForDistortionDetection)
├── train.py                  # Training script with DDP support
├── decode.py                 # Inference/evaluation script
├── dataset_hf_parquet.py     # HuggingFace Parquet dataset loader
├── archive_to_parquet.py     # Convert wav.scp + labels to parquet format
├── parquet_data.sh           # Batch data conversion script
├── train_ddp_hear_fail_deletion_only.sh  # Example training script (binary)
└── train_ddp_sound_event.sh  # Example training script (multi-class)
```

## Data Format

### Input: Parquet Files

The model expects data in Parquet format with the following schema:
- `uttid`: Utterance ID (string)
- `audio`: Audio bytes (bytes)
- `targets`: Dictionary containing:
  - `frame_labels`: Frame-level labels (numpy array, int64)
  - `word_labels`: Optional word-level labels
  - `word_timestamps`: Optional word timestamps

### Converting Data to Parquet

Use `archive_to_parquet.py` to convert Kaldi-style `wav.scp` and label files:

```bash
python archive_to_parquet.py \
    --wav_scp <path/to/wav.scp> \
    --labels <path/to/labels.txt> \
    --data_name <dataset_name> \
    --output_path <output_directory>
```

Or use the batch script `parquet_data.sh` (update paths first):

```bash
bash parquet_data.sh
```

## Training

### Single GPU Training

The model is trained on a single GPU with the following recommended settings:

```bash
python train.py \
    --train_data_dir <path/to/train/parquet> \
    --val_data_dir <path/to/val/parquet> \
    --output_dir <output_directory> \
    --batch_size 192 \
    --learning_rate 2e-4 \
    --num_epochs 40 \
    --num_labels 2 \
    --freeze_encoder \
    --freeze_preprocessor
```

### SLURM Job Submission

Example scripts are provided:
- `train_ddp_hear_fail_deletion_only.sh`: Binary classification (deletion detection)
- `train_ddp_sound_event.sh`: Multi-class classification (6 distortion types)

These scripts use single GPU training with batch size 192 and learning rate 2e-4. Update the paths in these scripts before use.

**Note:** The scripts are named with "ddp" but are configured for single GPU training. For multi-GPU training, use `torchrun` with `--nproc_per_node=N` where N is the number of GPUs.

### Training Parameters

**Data Arguments:**
- `--train_data_dir`: Directory containing training parquet files (required)
- `--val_data_dir`: Directory containing validation parquet files (required)
- `--max_length_seconds`: Maximum audio length in seconds (default: 30.0)
- `--num_workers`: Number of data loading workers (default: 4)

**Model Arguments:**
- `--model_name`: Parakeet model name (default: `nvidia/parakeet-tdt-0.6b-v2`)
- `--num_labels`: Number of classification labels (default: 2)
- `--freeze_encoder`: Freeze encoder weights (default: True)
- `--freeze_preprocessor`: Freeze preprocessor weights (default: True)
- `--projector_hidden_dim`: Hidden dimension for 2-layer projector (None for single layer)
- `--use_encoder_projection`: Use Parakeet's encoder projection (1024→640) before classifier
- `--cnn_kernel_size`: Kernel size for CNN classifier (None for linear)
- `--cnn_num_layers`: Number of CNN layers (default: 2)
- `--use_conv_transformer`: Use Conv-Transformer architecture
- `--conv_transformer_num_heads`: Number of attention heads (default: 8)
- `--conv_transformer_num_layers`: Number of Transformer layers (default: 3)

**Label Filtering:**
- `--remove_deletions`: Remove deletion errors from labels
- `--remove_sub`: Remove substitution errors from labels
- `--remove_ins`: Remove insertion errors from labels
- `--word_level`: Use word-level labels instead of frame-level

**Training Arguments:**
- `--batch_size`: Batch size (default: 4, recommended: 192 for single GPU)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 4)
- `--learning_rate`: Learning rate (default: 1e-4, recommended: 2e-4)
- `--num_epochs`: Number of training epochs (default: 10)
- `--warmup_steps`: Warmup steps for learning rate scheduler (default: 500)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: 10.0)
- `--log_interval`: Logging interval in steps (default: 10)
- `--eval_interval`: Evaluation interval in steps (default: 500)
- `--save_interval`: Checkpoint saving interval in steps (default: 500)
- `--resume_from_checkpoint`: Path to checkpoint to resume from

**Output Arguments:**
- `--output_dir`: Output directory for checkpoints and logs (default: `./output`)
- `--label_names`: Names for each label class (space-separated)

## Inference/Evaluation

Use `decode.py` to run inference on parquet data:

```bash
python decode.py \
    --checkpoint <path/to/checkpoint> \
    --data_dir <path/to/parquet/files> \
    --output_file <path/to/output.txt> \
    --num_labels 2 \
    --batch_size 32
```

### Decoding Parameters

**Required:**
- `--checkpoint`: Path to model checkpoint directory
- `--data_dir`: Directory containing parquet files to decode
- `--output_file`: Path to output text file

**Model Arguments:**
- `--num_labels`: Number of classification labels (default: 2)
- `--cnn_kernel_size`: Kernel size for CNN (must match training)
- `--cnn_num_layers`: Number of CNN layers (must match training)
- `--projector_hidden_dim`: Projector hidden dim (must match training)
- `--use_encoder_projection`: Use encoder projection (must match training)

**Decoding Arguments:**
- `--batch_size`: Batch size for decoding (default: 32)
- `--max_length_seconds`: Maximum audio length (default: 30.0)
- `--num_workers`: Number of data loading workers (default: 4)
- `--deletion_only`: Only decode deletion errors
- `--output_format`: Output format - `per_frame`, `summary`, or `both` (default: `per_frame`)
- `--no_metrics`: Skip evaluation metrics computation
- `--label_names`: Names for each label class

### Output Format

The decoder outputs:
- **Per-frame predictions**: Frame-level label predictions for each utterance
- **Summary statistics**: Accuracy, precision, recall, F1-score, confusion matrix
- **Metrics file**: Detailed metrics saved to `{output_file}.metrics.json`

## Model Checkpoints

Checkpoints are saved in the output directory with the following structure:

```
output_dir/
├── best_model/
│   ├── config.json              # Model configuration
│   ├── classifier.pt           # Classifier weights
│   ├── cnn.pt                   # CNN weights (if used)
│   └── parakeet_modules.pt      # Parakeet encoder/preprocessor weights
├── checkpoint_<step>/
│   └── ...
└── logs/                        # TensorBoard logs
```

## Label Types

### Binary Classification (num_labels=2)
- **0**: Clean / No error
- **1**: Distorted / Error present

### Multi-Class Classification (num_labels=6)
- **0**: Clean speech
- **1**: Noisy background
- **2**: Room Impulse Response (RIR)
- **3**: Interference speakers
- **4**: Network packet loss / low bitrate codec
- **5**: Totally missing segments

## Example Use Cases

### 1. Hearing Fault Detection (Binary)

Detect deletion errors in ASR output:

```bash
python train.py \
    --train_data_dir data/hearing_fault/train \
    --val_data_dir data/hearing_fault/valid \
    --batch_size 192 \
    --learning_rate 2e-4 \
    --num_labels 2 \
    --remove_sub \
    --remove_ins \
    --freeze_encoder \
    --freeze_preprocessor \
    --use_encoder_projection \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --output_dir output/hearing_fault
```

### 2. Sound Event Classification (Multi-Class)

Classify frame-level distortion types:

```bash
python train.py \
    --train_data_dir data/sound_event/train \
    --val_data_dir data/sound_event/valid \
    --batch_size 192 \
    --learning_rate 2e-4 \
    --num_labels 6 \
    --freeze_encoder \
    --freeze_preprocessor \
    --use_encoder_projection \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --output_dir output/sound_event
```

## Dependencies

- Python 3.7+
- PyTorch >= 1.10.0
- NeMo (for Parakeet models)
- HuggingFace Datasets
- PyArrow (for Parquet support)
- NumPy
- SciPy
- SoundFile
- scikit-learn
- tqdm
- TensorBoard

## Notes

- The Parakeet encoder and preprocessor are frozen by default to preserve pre-trained features
- Frame labels are expected at 50 Hz (20ms frame hop)
- Audio is automatically resampled to 16 kHz if needed
- The model supports variable-length sequences via padding
- Recommended training configuration: single GPU, batch size 192, learning rate 2e-4
- For multi-GPU training, use `torchrun` with `--nproc_per_node=N`
- Checkpoints are saved at regular intervals and for the best validation accuracy

## Citation

If you use this model, please cite:
- Parakeet TDT model (NVIDIA)
- NeMo toolkit
- Relevant datasets used for training

