#!/usr/bin/env bash

#SBATCH -o log/decode_response.log
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -w <NODE_NAME>

audio_path=$1
result_path=$2
mkdir -p $result_path
./Parakeet-ASR/inference_with_joint_emb_logits.py --audio_dir $audio_path --result_dir $result_path

