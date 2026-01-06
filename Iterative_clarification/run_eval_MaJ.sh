#!/usr/bin/env bash

# Configuration - update these paths
# Dataset names
DATASET1="<DATASET1>"
DATASET2="<DATASET2>"

# Paths
ASR_RESULTS_DIR="<PATH_TO_ASR_RESULTS>"
DATA_DIR="<PATH_TO_DATA>"

# Process Dataset 1: Generate dialogue responses from ASR
python utils/dialogue_gpt.py \
    --text ${ASR_RESULTS_DIR}/${DATASET1}_asr_joint/recog.txt \
    --output ${DATA_DIR}/${DATASET1}/result.asr.txt

# Process Dataset 2: Generate dialogue responses from ASR
python utils/dialogue_gpt.py \
    --text ${ASR_RESULTS_DIR}/${DATASET2}_asr_joint/recog.txt \
    --output ${DATA_DIR}/${DATASET2}/result.asr.txt

# Evaluate Dataset 2: Scale scoring (0-5)
python utils/run_evaluation.py \
    --text ${DATA_DIR}/${DATASET2}/text \
    --reference ${DATA_DIR}/${DATASET2}/answer.txt \
    --prediction ${DATA_DIR}/${DATASET2}/result.asr.txt \
    --output ${DATA_DIR}/${DATASET2}/evaluation_results.asr.json \
    --mode scale

# Evaluate Dataset 1: Scale scoring (0-5)
python utils/run_evaluation.py \
    --text ${DATA_DIR}/${DATASET1}/text \
    --reference ${DATA_DIR}/${DATASET1}/answer.txt \
    --prediction ${DATA_DIR}/${DATASET1}/result.asr.txt \
    --output ${DATA_DIR}/${DATASET1}/evaluation_results.asr.json \
    --mode scale