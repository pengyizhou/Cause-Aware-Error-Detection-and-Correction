#!/usr/bin/env bash

#SBATCH --job-name=parakeet_decode_confidence
#SBATCH --gres=gpu:1
#SBATCH -w <NODE_NAME>
#SBATCH --cpus-per-task=4
#SBATCH -o ./log/parakeet_decode_confidence.log

# Configuration - update these paths and FPR values

# Process additional dataset partitions
parts="test"
src="<PATH_TO_DATASET>"

for part in $parts; do
    echo "Decoding $part"
    audio_path="$src/$part/wav.scp"
    result_path="./results/<DATASET_NAME>/$part"

    [ -d $result_path ] || mkdir -p $result_path

    ./inference_with_joint_emb_logits.py \
        --audio_scp $audio_path \
        --result_dir $result_path

    python evaluate_confidence_recall.py \
        --confidence_json $result_path/confidence.json \
        --hyp_file $result_path/recog.txt \
        --ref_file $result_path/transcription.txt \
        --target_fpr 0.0113
done
