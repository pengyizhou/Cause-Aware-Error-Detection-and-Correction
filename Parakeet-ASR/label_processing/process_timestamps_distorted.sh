#!/usr/bin/env bash

# Configuration - update these paths
datasets="<DATASET1> <DATASET2> <DATASET3>"
parts="interference missing multi_dist_no_rir multi_dist_rir noise noise_partial packet_loss rir rir_noise"

for dataset in $datasets; do
    timestamp_clean_file="<PATH_TO_RESULTS>/$dataset/all_clean/token_timestamps.txt.nooverlap"
    sort -k1 -o $timestamp_clean_file $timestamp_clean_file
    for part in $parts; do
        timestamp_distorted_file="<PATH_TO_RESULTS>/$dataset/$part/token_timestamps.txt.nooverlap"
        if [ -f $timestamp_distorted_file ]; then
            echo "Processing $dataset/$part"
            sort -k1 -o $timestamp_distorted_file $timestamp_distorted_file
            cut -f1 $timestamp_distorted_file > $timestamp_distorted_file.uttlist
            grep -F -f $timestamp_distorted_file.uttlist $timestamp_clean_file > $timestamp_distorted_file.clean.txt
            cut -f1 $timestamp_distorted_file.clean.txt > $timestamp_distorted_file.clean.uttlist
            grep -F -f $timestamp_distorted_file.clean.uttlist $timestamp_distorted_file > $timestamp_distorted_file.final.txt
        else
            echo "Timestamp file not found for $dataset/$part"
        fi
    done
done