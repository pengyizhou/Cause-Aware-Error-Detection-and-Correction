#!/usr/bin/env bash

# Script to convert joint embeddings and labels to parquet format
# Usage: Update the variables below and uncomment the relevant section

# Configuration variables - update these for your setup
# BASE_DATA_DIR="path/to/your/data"
# BASE_OUTPUT_DIR="path/to/output"
# ASR_RESULTS_DIR="path/to/asr/results"

# Example: Process dataset with joint embeddings
dataset="AESRC2020"
parts="valid test train"
save_dir="${BASE_OUTPUT_DIR}/${dataset}_joint_embeddings"

for part in $parts; do
    embeddings_file="${ASR_RESULTS_DIR}/${dataset}/${part}/joint_embeddings.pt"
    labels_file="${ASR_RESULTS_DIR}/${dataset}/${part}/token_timestamps.labels.txt"
    python ./archive_to_parquet.py \
        --embeddings $embeddings_file \
        --labels $labels_file \
        --data_name $dataset-$part \
        --output_path $save_dir
done
