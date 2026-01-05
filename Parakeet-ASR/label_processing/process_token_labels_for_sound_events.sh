#!/usr/bin/env bash

# Configuration - update these paths
datasets="<DATASET1> <DATASET2> <DATASET3>"
parts="interference missing multi_dist_no_rir multi_dist_rir noise noise_partial packet_loss rir rir_noise"

for dataset in $datasets; do
    for part in $parts; do
        label_dir="<PATH_TO_LABEL_DATA>/$dataset/distorted/$part/$part"
        output_file="<PATH_TO_RESULTS>/$dataset/$part/token_timestamps.txt.nooverlap.labels.sound_events.txt"
        python3 ./generate_labels_with_npy_labels.py $label_dir $output_file
    done
done
