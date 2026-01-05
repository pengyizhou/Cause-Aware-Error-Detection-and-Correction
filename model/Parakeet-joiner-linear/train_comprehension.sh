#!/usr/bin/env bash

#SBATCH -o log/train_joint_embeddings_linear_all_understanding_cnn_classifier_layer5.log
#SBATCH -w <NODE_NAME>
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --dependency=afterany:<JOB_ID>

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}
# NUM_GPUS=1
echo "Starting training with $NUM_GPUS GPUs using torchrun..."

# Use torchrun (recommended for PyTorch 1.10+)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=30510 \
    ./train.py \
    --train_data_dir <PATH_TO_TRAIN_DATA> \
    --val_data_dir <PATH_TO_VAL_DATA> \
    --num_labels 2 \
    --cnn_kernel_size 5 \
    --cnn_num_layers 5 \
    --dropout_prob 0.2 \
    --learning_rate 2e-4 \
    --output_dir <PATH_TO_OUTPUT_DIR> \
    --num_epochs 40 \
    --batch_size 192 \
    --gradient_accumulation_steps 1