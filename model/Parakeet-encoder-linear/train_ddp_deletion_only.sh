#!/bin/bash

#SBATCH -o log/train_ddp_two_layers_hearing_fault_with_encoder_projection_640_cnn_classifier_layer5_deletion_only.log
#SBATCH -w <NODE_NAME>
#SBATCH --gres=gpu:1
#SBATCH -c 4




# Distributed training launcher script for HuBERT distortion detection
# This script launches training with DDP using torchrun



# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
# NUM_GPUS=1
echo "Starting distributed training with $NUM_GPUS GPUs using torchrun..."

# Use torchrun (recommended for PyTorch 1.10+)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=30507 \
    ./train.py \
        --train_data_dir <PATH_TO_TRAIN_DATA> \
        --val_data_dir <PATH_TO_VAL_DATA> \
        --batch_size 192 \
        --gradient_accumulation_steps 1 \
        --learning_rate 4e-4 \
        --num_epochs 40 \
        --output_dir <PATH_TO_OUTPUT_DIR> \
        --projector_hidden_dim 640 \
        --model_name nvidia/parakeet-tdt-0.6b-v2 \
        --num_labels 2 \
        --remove_sub \
        --remove_ins \
        --freeze_encoder \
        --freeze_preprocessor \
        --use_encoder_projection \
        --cnn_kernel_size 5 \
        --cnn_num_layers 5
