#!/usr/bin/env bash

# Configuration - update these paths
datasets="<DATASET1> <DATASET2> <DATASET3>"
parts="interference missing multi_dist_no_rir multi_dist_rir noise noise_partial packet_loss rir rir_noise"

for dataset in $datasets; do
    for part in $parts; do
        timestamp_distorted_file="<PATH_TO_RESULTS>/$dataset/$part/token_timestamps.txt.nooverlap"
        clean_file="$timestamp_distorted_file.clean.txt"
        final_file="$timestamp_distorted_file.final.txt"
        output_file="$timestamp_distorted_file.labels.deletion2.norm.txt"
        
        if [ -f "$clean_file" ] && [ -f "$final_file" ]; then
            echo "Processing $dataset/$part"
            python3 ./compare_timestamps_CSID.py "$clean_file" "$final_file" "$output_file"
        else
            if [ ! -f "$clean_file" ]; then
                echo "Clean file not found for $dataset/$part: $clean_file"
            fi
            if [ ! -f "$final_file" ]; then
                echo "Final file not found for $dataset/$part: $final_file"
            fi
        fi
    done
done

