#!/usr/bin/env bash

#SBATCH --job-name=parakeet_decode_accent
#SBATCH --gres=gpu:1
#SBATCH -w <NODE_NAME>
#SBATCH --cpus-per-task=4
#SBATCH -o ./log/parakeet_decode_accent_logits.log

# Configuration - update these paths
parts="valid test train"
src="<PATH_TO_DATASET>"

for part in $parts; do
    echo "Decoding $part"
    audio_path="$src/$part/wav.scp"
    result_path="Parakeet-ASR/results/<DATASET_NAME>/$part"

    [ -d $result_path ] || mkdir -p $result_path

    ./inference_with_joint_emb_logits.py --audio_scp $audio_path --result_dir $result_path

    # Optional: Sort and compute WER
    sort $result_path/recog.txt -o $result_path/recog.txt
    sort $src/$part/text -o $result_path/transcription.txt
    ./compute-wer.py --char=1 --v=1 \
        $result_path/transcription.txt $result_path/recog.txt > $result_path/wer_result.txt
done
