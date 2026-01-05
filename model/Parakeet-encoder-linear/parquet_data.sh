#!/usr/bin/env bash

# Script to convert audio data from wav.scp format to parquet format
# Usage: Update the variables below and uncomment the relevant section

# Configuration variables - update these for your setup
# BASE_DATA_DIR="path/to/your/data"
# BASE_OUTPUT_DIR="path/to/output"
# ASR_RESULTS_DIR="path/to/asr/results"

# Example 1: Process dataset with standard labels
dataset="AESRC2020"
parts="test valid train"
save_dir="${BASE_OUTPUT_DIR}/${dataset}"

for part in $parts; do
    basedir="${BASE_DATA_DIR}/${dataset}/${part}"
    labels_file="${ASR_RESULTS_DIR}/${dataset}/${part}/token_timestamps.labels.txt"
    python ./archive_to_parquet.py \
        --wav_scp $basedir/wav.scp \
        --labels $labels_file \
        --data_name $dataset-$part \
        --output_path $save_dir
done
