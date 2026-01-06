#!/usr/bin/env bash

#SBATCH --job-name=tts_generation
#SBATCH --gres=gpu:1
#SBATCH -w <NODE_NAME>
#SBATCH --cpus-per-task=4
#SBATCH -o ./log/tts_generation.log

# Configuration - update these paths
# Reference audio folder containing .wav and .txt pairs
REF_FOLDER="<PATH_TO_REFERENCE_AUDIO>"

# TTS input file (format: uttid text)
TTS_INPUT_FILE="<PATH_TO_TTS_INPUT>"

# Output directory for generated audio files
OUTPUT_DIR="<PATH_TO_OUTPUT>"

# Optional: CosyVoice model directory (defaults to placeholder if not set)
MODEL_DIR="${MODEL_DIR:-<PATH_TO_PRETRAINED_MODELS>/Cosyvoice3-0.5B}"

# Optional: Maximum words in reference text (default: 10)
MAX_WORDS="${MAX_WORDS:-10}"

# Optional: Third-party path for Matcha-TTS (defaults to placeholder if not set)
export THIRD_PARTY_PATH="${THIRD_PARTY_PATH:-<PATH_TO_THIRD_PARTY>}"

# Run TTS generation
python3 ./utils/gen_rounds.py \
    --ref_folder "${REF_FOLDER}" \
    --tts_input_file "${TTS_INPUT_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_dir "${MODEL_DIR}" \
    --max_words "${MAX_WORDS}"

