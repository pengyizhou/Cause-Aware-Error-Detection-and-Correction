#!/usr/bin/env bash

#SBATCH --job-name=classifier_eval
#SBATCH --gres=gpu:1
#SBATCH -w <NODE_NAME>
#SBATCH --cpus-per-task=4
#SBATCH -o ./log/classifier_eval.log

# Configuration - update these paths
datasets=("<DATASET1>" "<DATASET2>")

for dataset in ${datasets[@]}; do
    # 1. Sound Event Detection (6-class distortion classification)
    python ./Parakeet-encoder-linear/decode.py \
        --data_dir <PATH_TO_DATA>/${dataset}/parquet_encoder/${dataset} \
        --output_file <PATH_TO_OUTPUT>/${dataset}/result_encoder/event.txt \
        --checkpoint ./Parakeet-encoder-linear/averaged_model_sound_event \
        --projector_hidden_dim 640 \
        --use_encoder_projection \
        --cnn_kernel_size 5 \
        --cnn_num_layers 5 \
        --num_labels 6

    # 2. Deletion Error Detection (2-class: clean vs deletion)
    python ./Parakeet-encoder-linear/decode.py \
        --data_dir <PATH_TO_DATA>/${dataset}/parquet_encoder/${dataset} \
        --output_file <PATH_TO_OUTPUT>/${dataset}/result_encoder/deletion.txt \
        --checkpoint ./Parakeet-encoder-linear/averaged_model_deletion_only \
        --projector_hidden_dim 640 \
        --use_encoder_projection \
        --cnn_kernel_size 5 \
        --cnn_num_layers 5 \
        --num_labels 2

    # 3. Comprehension/Understanding Detection (2-class: intelligible vs unintelligible)
    python ./Parakeet-joiner-linear/decode.py \
        --data_dir <PATH_TO_DATA>/${dataset}/parquet_joint/${dataset} \
        --output_file <PATH_TO_OUTPUT>/${dataset}/result_joint/understanding.txt \
        --checkpoint ./Parakeet-joiner-linear/averaged_model_comprehension \
        --cnn_kernel_size 5 \
        --cnn_num_layers 5 \
        --num_labels 2

    # 4. Perception/Hearing Fault Detection (2-class: clean vs error)
    python ./Parakeet-joiner-linear/decode.py \
        --data_dir <PATH_TO_DATA>/${dataset}/parquet_joint/${dataset} \
        --output_file <PATH_TO_OUTPUT>/${dataset}/result_joint/hearing_fault.txt \
        --checkpoint ./Parakeet-joiner-linear/averaged_model_perception \
        --cnn_kernel_size 5 \
        --cnn_num_layers 5 \
        --num_labels 2

    # 5. Evaluate word-level errors with all predictions
    python3 ./evaluate_word_level_errors.py \
        --transcription <PATH_TO_DATA>/${dataset}/text \
        --recog <PATH_TO_ASR_RESULTS>/dialogue/${dataset}/recog.txt \
        --token-timestamps <PATH_TO_ASR_RESULTS>/dialogue/${dataset}/token_timestamps.txt \
        --intelligibility-predictions <PATH_TO_OUTPUT>/${dataset}/result_joint/understanding.jsonl \
        --hearing-fault-predictions <PATH_TO_OUTPUT>/${dataset}/result_joint/hearing_fault.jsonl \
        --deletion-predictions <PATH_TO_OUTPUT>/${dataset}/result_encoder/deletion.jsonl \
        --output <PATH_TO_OUTPUT>/${dataset}/result_joint/understanding_marked.jsonl \
        --output-metrics <PATH_TO_OUTPUT>/${dataset}/result_joint/understanding_metrics.json
done