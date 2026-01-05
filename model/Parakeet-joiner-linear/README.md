# Parakeet-Joiner-Linear: Frame-Level Distortion Detection from Joint Embeddings

This directory contains a PyTorch implementation for frame-level speech distortion detection using pre-computed Parakeet joint embeddings (640-dimensional) with a linear/CNN classification head.

## Overview

The model operates on pre-extracted joint embeddings from NVIDIA's Parakeet TDT model (640-dimensional features from the joint network). It applies a trainable classification head to predict distortion types at 50 Hz (20ms frame hop). The model supports both binary classification (e.g., intelligibility detection) and multi-class classification (e.g., distortion type classification).

## Model Architecture

### Input Features
- **Joint Embeddings**: 640-dimensional frame-level features extracted from Parakeet's joint network
- **Frame Rate**: 50 Hz (20ms frame hop)
- **Input Shape**: `[batch, time, 640]`

### Classification Head Options

1. **Linear Classifier** (default)
   - Architecture: `ReLU → Dropout(0.2) → Linear(640, num_labels)`
   - Simple and efficient for binary classification

2. **Two-Layer Projector + Linear**
   - Optional hidden projection layer: `640 → hidden_dim → num_labels`
   - Set `--projection_dim` to enable
   - Recommended values: 128, 256, or 512

3. **CNN Classifier**
   - Convolutional layers for local window processing
   - Configurable kernel size and number of layers
   - Set `--cnn_kernel_size` and `--cnn_num_layers` to enable
   - Architecture: `Conv1d → BatchNorm → ReLU → Dropout` (repeated)

4. **Conv-Transformer**
   - 2 CNN layers + Transformer encoder layers + linear classifier
   - Set `--use_conv_transformer` to enable
   - Architecture: `Conv1d x 2 → Transformer x N → Linear`

## Directory Structure

```
Parakeet-joiner-linear/
├── README.md                 # This file
├── model.py                  # Model definition (ParakeetForDistortionDetection)
├── train.py                  # Training script with DDP support
├── decode.py                 # Inference/evaluation script
├── dataset_hf_parquet.py     # HuggingFace Parquet dataset loader
├── archive_to_parquet.py     # Convert joint embeddings + labels to parquet format
├── parquet_data.sh           # Batch data conversion script
├── train_comprehension.sh    # Example training script (comprehension task)
└── train_perception.sh       # Example training script (perception task)
```

## Data Format

### Input: Parquet Files

The model expects data in Parquet format with the following schema:
- `uttid`: Utterance ID (string)
- `embedding`: Joint embedding array `[time, 640]` (numpy array, float32)
- `targets`: Dictionary containing:
  - `frame_labels`: Frame-level labels (numpy array, int64)

### Converting Data to Parquet

Use `archive_to_parquet.py` to convert joint embeddings (`.pt` file) and label files:

```bash
python archive_to_parquet.py \
    --embeddings <path/to/joint_embeddings.pt> \
    --labels <path/to/labels.txt> \
    --data_name <dataset_name> \
    --output_path <output_directory>
```

The script expects:
- **Embeddings file**: A `.pt` file containing a dictionary mapping `uttid → Tensor[time, 640]`
- **Labels file**: A text file with frame-level labels (one per line or space-separated per utterance)

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
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --dropout_prob 0.2
```

### SLURM Job Submission

Example scripts are provided:
- `train_comprehension.sh`: Training for comprehension/intelligibility tasks
- `train_perception.sh`: Training for perception tasks

These scripts use single GPU training with batch size 192 and learning rate 2e-4. Update the paths in these scripts before use.

**Note:** The scripts use `torchrun` but are configured for single GPU training. For multi-GPU training, use `torchrun` with `--nproc_per_node=N` where N is the number of GPUs.

### Training Parameters

**Data Arguments:**
- `--train_data_dir`: Directory containing training parquet files (required)
- `--val_data_dir`: Directory containing validation parquet files (required)
- `--max_length_seconds`: Maximum sequence length in seconds (default: 30.0)
- `--num_workers`: Number of data loading workers (default: 4)

**Model Arguments:**
- `--input_dim`: Input feature dimension (default: 640)
- `--num_labels`: Number of classification labels (default: 2)
- `--dropout_prob`: Dropout probability (default: 0.2)
- `--projection_dim`: Hidden dimension for intermediate layer (None for direct classification)
- `--num_linear_layers`: Number of linear layers (default: None, uses 1 layer)
- `--cnn_kernel_size`: Kernel size for CNN classifier (None for linear)
- `--cnn_num_layers`: Number of CNN layers (default: 2)
- `--use_conv_transformer`: Use Conv-Transformer architecture
- `--conv_transformer_num_heads`: Number of attention heads (default: 8)
- `--conv_transformer_num_layers`: Number of Transformer layers (default: 3)

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

Use `decode.py` to run inference on parquet data or joint embeddings:

### Inference on Parquet Files

```bash
python decode.py \
    --checkpoint <path/to/checkpoint> \
    --data_dir <path/to/parquet/files> \
    --output_file <path/to/output.txt> \
    --num_labels 2 \
    --batch_size 32
```

### Inference on Joint Embeddings (.pt file)

```bash
python decode.py \
    --checkpoint <path/to/checkpoint> \
    --embeddings_pt <path/to/joint_embeddings.pt> \
    --output_file <path/to/output.txt> \
    --num_labels 2 \
    --batch_size 32
```

### Decoding Parameters

**Required:**
- `--checkpoint`: Path to model checkpoint directory
- Either `--data_dir` (parquet files) or `--embeddings_pt` (joint embeddings file)
- `--output_file`: Path to output text file

**Model Arguments:**
- `--num_labels`: Number of classification labels (default: 2)
- `--input_dim`: Input feature dimension (default: 640, must match training)
- `--cnn_kernel_size`: Kernel size for CNN (must match training)
- `--cnn_num_layers`: Number of CNN layers (must match training)
- `--dropout_prob`: Dropout probability (must match training)
- `--projection_dim`: Projection hidden dim (must match training)
- `--use_conv_transformer`: Use Conv-Transformer (must match training)

**Decoding Arguments:**
- `--batch_size`: Batch size for decoding (default: 32)
- `--max_length_frames`: Maximum sequence length in frames (default: None)
- `--num_workers`: Number of data loading workers (default: 4)
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
│   └── classifier.pt           # Classifier weights
├── checkpoint_<step>/
│   └── ...
└── logs/                        # TensorBoard logs
```

## Label Types

### Binary Classification (num_labels=2)
- **0**: Clean / No error / Intelligible
- **1**: Distorted / Error present / Unintelligible

### Multi-Class Classification (num_labels=6)
- **0**: Clean speech
- **1**: Noisy background
- **2**: Room Impulse Response (RIR)
- **3**: Interference speakers
- **4**: Network packet loss / low bitrate codec
- **5**: Totally missing segments

## Example Use Cases

### 1. Intelligibility Detection (Binary)

Detect intelligibility errors in ASR output:

```bash
python train.py \
    --train_data_dir data/intelligibility/train \
    --val_data_dir data/intelligibility/valid \
    --batch_size 192 \
    --learning_rate 2e-4 \
    --num_labels 2 \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --dropout_prob 0.2 \
    --output_dir output/intelligibility
```

### 2. Comprehension Task (Binary)

Classify comprehension-related errors:

```bash
python train.py \
    --train_data_dir data/comprehension/train \
    --val_data_dir data/comprehension/valid \
    --batch_size 192 \
    --learning_rate 2e-4 \
    --num_labels 2 \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --dropout_prob 0.2 \
    --output_dir output/comprehension
```

### 3. Perception Task (Binary)

Classify perception-related errors:

```bash
python train.py \
    --train_data_dir data/perception/train \
    --val_data_dir data/perception/valid \
    --batch_size 192 \
    --learning_rate 2e-4 \
    --num_labels 2 \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --dropout_prob 0.2 \
    --output_dir output/perception
```

## Extracting Joint Embeddings

Before training, you need to extract joint embeddings from Parakeet. This is typically done using a script like `Parakeet-ASR/inference_with_joint_emb.py`:

```bash
python inference_with_joint_emb.py \
    --wav_scp <path/to/wav.scp> \
    --output_file joint_embeddings.pt
```

The output `.pt` file should contain a dictionary:
```python
{
    "uttid1": Tensor[time, 640],
    "uttid2": Tensor[time, 640],
    ...
}
```

## Dependencies

- Python 3.7+
- PyTorch >= 1.10.0
- HuggingFace Datasets
- PyArrow (for Parquet support)
- NumPy
- scikit-learn
- tqdm
- TensorBoard

## Notes

- The model operates on pre-computed joint embeddings (640-dim) from Parakeet's joint network
- Frame labels are expected at 50 Hz (20ms frame hop)
- The embedding time dimension must match the label length for each utterance
- Recommended training configuration: single GPU, batch size 192, learning rate 2e-4
- For multi-GPU training, use `torchrun` with `--nproc_per_node=N`
- Checkpoints are saved at regular intervals and for the best validation accuracy
- The model supports variable-length sequences via padding

## Differences from Parakeet-Encoder-Linear

This model differs from `Parakeet-encoder-linear` in that:
- **Input**: Pre-computed joint embeddings (640-dim) instead of raw audio
- **Processing**: No audio preprocessing or encoder forward pass needed
- **Efficiency**: Faster training and inference (no encoder computation)
- **Use Case**: Best when joint embeddings are already available from ASR inference

## Citation

If you use this model, please cite:
- Parakeet TDT model (NVIDIA)
- NeMo toolkit
- Relevant datasets used for training

