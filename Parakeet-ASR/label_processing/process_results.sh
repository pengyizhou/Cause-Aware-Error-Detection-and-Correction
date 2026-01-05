#!/usr/bin/env bash

# Configuration - update these paths
datasets="<DATASET1> <DATASET2> <DATASET3>"
parts="all_clean interference missing multi_dist_no_rir multi_dist_rir noise noise_partial packet_loss rir rir_noise"

for dataset in $datasets; do
    for part in $parts; do
        timestamp_file="<PATH_TO_RESULTS>/$dataset/$part/token_timestamps.txt"
        if [ -f $timestamp_file ]; then
            echo "Processing $dataset/$part"
            grep -v "\[\]" $timestamp_file > $timestamp_file.clean

            ./cleanup_token_timestamps.py -i $timestamp_file.clean -o $timestamp_file.nooverlap
        else
            echo "Timestamp file not found for $dataset/$part"
        fi
    done
done