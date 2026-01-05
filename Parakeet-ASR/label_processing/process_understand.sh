#!/usr/bin/env bash

#SBATCH --dependency=afterany:<JOB_ID>
#SBATCH -o log/process_spgi_understand.log

# Configuration - update these paths
datasets="<DATASET1> <DATASET2>"
parts=("all_clean")

for dataset in $datasets; do
    for part in $parts; do
        basedir="<PATH_TO_RESULTS>/$dataset/$part"

        python ./generate_token_labels.py \
            --timestamps $basedir/timestamps.txt \
            --token_timestamps $basedir/token_timestamps.txt \
            --transcription $basedir/transcription.txt \
            --output $basedir/token_timestamps.labels.txt \
            --deletion_strategy skip
    done
done