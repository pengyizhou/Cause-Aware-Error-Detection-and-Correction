#!/usr/bin/env bash

# Configuration - update these paths and dataset names
# Dataset names (used as prefixes)
DATASET1_PREFIX="<DATASET1_PREFIX>"
DATASET2_PREFIX="<DATASET2_PREFIX>"

# Base paths
DATA_DIR="<PATH_TO_DATA>"
CLARIFICATION_DIR="<PATH_TO_CLARIFICATION>"
TTS_DIR="<PATH_TO_TTS>"

# Initial: Prepare initial ASR with error markers
cat ${DATA_DIR}/${DATASET1_PREFIX}/result_joint/understanding_marked.txt | awk -v prefix="${DATASET1_PREFIX}" '{ print prefix"_"$0}' > ${CLARIFICATION_DIR}/round_1_recog.txt

cat ${DATA_DIR}/${DATASET2_PREFIX}/result_joint/understanding_marked.txt | awk -v prefix="${DATASET2_PREFIX}" '{ print prefix"_"$0}' >> ${CLARIFICATION_DIR}/round_1_recog.txt

python ./utils/prepare_batch_dialogue.py \
    ${CLARIFICATION_DIR}/round_1_recog.txt \
    -o ${CLARIFICATION_DIR}/round_1_dialogue.batch.jsonl \
    --round 1 \
    --original-asr ${CLARIFICATION_DIR}/round_1_recog.txt

python ./utils/create_batch_infer.py ${CLARIFICATION_DIR}/round_1_dialogue.batch.jsonl ${CLARIFICATION_DIR}/round_1_dialogue_response.json

python ./utils/extract_text_from_dialogue.py ${CLARIFICATION_DIR}/round_1_dialogue_response.json ${CLARIFICATION_DIR}/round_1_to_round2_clarification_questions.txt ${CLARIFICATION_DIR}/round_1_final_asr_fixed.txt

python3 ./utils/split_finalized_asr.py \
  --input ${CLARIFICATION_DIR}/round_1_final_asr_fixed.txt \
  --finalized ${CLARIFICATION_DIR}/round_1_final_asr_fixed.finalized.txt \
  --not-finalized ${CLARIFICATION_DIR}/round_1_final_asr_fixed.not_finalized.txt

# Prepare reference text
cat ${DATA_DIR}/${DATASET1_PREFIX}/text | awk -v prefix="${DATASET1_PREFIX}" '{ print prefix"_"$0}' > ${CLARIFICATION_DIR}/round_1_reference.txt

cat ${DATA_DIR}/${DATASET2_PREFIX}/text | awk -v prefix="${DATASET2_PREFIX}" '{ print prefix"_"$0}' >> ${CLARIFICATION_DIR}/round_1_reference.txt

python3 ./utils/prepare_batch_user_response.py \
  --reference ${CLARIFICATION_DIR}/round_1_reference.txt \
  --clarification ${CLARIFICATION_DIR}/round_1_to_round2_clarification_questions.txt \
  --original-asr ${CLARIFICATION_DIR}/round_1_final_asr_fixed.txt \
  --output ${CLARIFICATION_DIR}/round_1_user_response.batch.jsonl

python3 ./utils/create_batch_infer.py ${CLARIFICATION_DIR}/round_1_user_response.batch.jsonl ${CLARIFICATION_DIR}/round_1_user_response.responses.jsonl

python3 ./utils/extract_user_response_direct.py \
  --input ${CLARIFICATION_DIR}/round_1_user_response.responses.jsonl \
  --output ${CLARIFICATION_DIR}/round_1_user_response.tts.txt \
  --round 1

python3 ./utils/normalize_text_for_tts.py \
    --input ${CLARIFICATION_DIR}/round_1_user_response.tts.txt \
    --output ${CLARIFICATION_DIR}/round_1_user_response.tts.txt.norm

# Recognize the audio generated from the TTS
sbatch ./utils/decode_response.sh ${TTS_DIR}/instruction_round1_gen ${CLARIFICATION_DIR}/round_1_user_response

sort -o ${CLARIFICATION_DIR}/round_1_user_response/recog.txt ${CLARIFICATION_DIR}/round_1_user_response/recog.txt

# Round 1: Continue clarification for non-finalized utterances
python ./utils/prepare_batch_dialogue.py \
  ${CLARIFICATION_DIR}/round_1_final_asr_fixed.not_finalized.txt \
  -o ${CLARIFICATION_DIR}/round_2_dialogue.batch.jsonl \
  --round 2 \
  --round1-questions ${CLARIFICATION_DIR}/round_1_to_round2_clarification_questions.txt \
  --round1-answers ${CLARIFICATION_DIR}/round_1_user_response/recog.txt \
  --original-asr ${CLARIFICATION_DIR}/round_1_final_asr_fixed.not_finalized.txt

python ./utils/create_batch_infer.py ${CLARIFICATION_DIR}/round_2_dialogue.batch.jsonl ${CLARIFICATION_DIR}/round_2_dialogue_response.json

python ./utils/extract_text_from_dialogue.py \
  ${CLARIFICATION_DIR}/round_2_dialogue_response.json \
  ${CLARIFICATION_DIR}/round_2_to_round3_clarification_questions.txt \
  ${CLARIFICATION_DIR}/round_2_final_asr_fixed.txt

python3 ./utils/split_finalized_asr.py \
  --input ${CLARIFICATION_DIR}/round_2_final_asr_fixed.txt \
  --finalized ${CLARIFICATION_DIR}/round_2_final_asr_fixed.finalized.txt \
  --not-finalized ${CLARIFICATION_DIR}/round_2_final_asr_fixed.not_finalized.txt

python3 ./utils/prepare_batch_user_response.py \
  --reference ${CLARIFICATION_DIR}/round_1_reference.txt \
  --clarification ${CLARIFICATION_DIR}/round_2_to_round3_clarification_questions.txt \
  --original-asr ${CLARIFICATION_DIR}/round_2_final_asr_fixed.not_finalized.txt \
  --output ${CLARIFICATION_DIR}/round_2_user_response.batch.jsonl

python3 ./utils/create_batch_infer.py \
  ${CLARIFICATION_DIR}/round_2_user_response.batch.jsonl \
  ${CLARIFICATION_DIR}/round_2_user_response.responses.jsonl

python3 ./utils/extract_user_response_direct.py \
  --input ${CLARIFICATION_DIR}/round_2_user_response.responses.jsonl \
  --output ${CLARIFICATION_DIR}/round_2_user_response.tts.txt \
  --round 2

python3 ./utils/normalize_text_for_tts.py \
    --input ${CLARIFICATION_DIR}/round_2_user_response.tts.txt \
    --output ${CLARIFICATION_DIR}/round_2_user_response.tts.txt.norm

# Recognize the audio generated from the TTS
sbatch ./utils/decode_response.sh ${TTS_DIR}/instruction_round2_gen ${CLARIFICATION_DIR}/round_2_user_response

sort -o ${CLARIFICATION_DIR}/round_2_user_response/recog.txt ${CLARIFICATION_DIR}/round_2_user_response/recog.txt

# Round 2: Continue clarification for non-finalized utterances
python ./utils/prepare_batch_dialogue.py \
  ${CLARIFICATION_DIR}/round_2_final_asr_fixed.not_finalized.txt \
  -o ${CLARIFICATION_DIR}/round_3_dialogue.batch.jsonl \
  --round 3 \
  --round1-questions ${CLARIFICATION_DIR}/round_2_to_round3_clarification_questions.txt \
  --round1-answers ${CLARIFICATION_DIR}/round_2_user_response/recog.txt \
  --original-asr ${CLARIFICATION_DIR}/round_2_final_asr_fixed.not_finalized.txt

python ./utils/create_batch_infer.py \
    ${CLARIFICATION_DIR}/round_3_dialogue.batch.jsonl \
    ${CLARIFICATION_DIR}/round_3_dialogue_response.json

python ./utils/extract_text_from_dialogue.py \
  ${CLARIFICATION_DIR}/round_3_dialogue_response.json \
  ${CLARIFICATION_DIR}/round_3_to_round4_clarification_questions.txt \
  ${CLARIFICATION_DIR}/round_3_final_asr_fixed.txt

python3 ./utils/split_finalized_asr.py \
  --input ${CLARIFICATION_DIR}/round_3_final_asr_fixed.txt \
  --finalized ${CLARIFICATION_DIR}/round_3_final_asr_fixed.finalized.txt \
  --not-finalized ${CLARIFICATION_DIR}/round_3_final_asr_fixed.not_finalized.txt

python3 ./utils/prepare_batch_user_response.py \
  --reference ${CLARIFICATION_DIR}/round_1_reference.txt \
  --clarification ${CLARIFICATION_DIR}/round_3_to_round4_clarification_questions.txt \
  --original-asr ${CLARIFICATION_DIR}/round_3_final_asr_fixed.not_finalized.txt \
  --output ${CLARIFICATION_DIR}/round_3_user_response.batch.jsonl

python3 ./utils/create_batch_infer.py \
  ${CLARIFICATION_DIR}/round_3_user_response.batch.jsonl \
  ${CLARIFICATION_DIR}/round_3_user_response.responses.jsonl

python3 ./utils/extract_user_response_direct.py \
  --input ${CLARIFICATION_DIR}/round_3_user_response.responses.jsonl \
  --output ${CLARIFICATION_DIR}/round_3_user_response.tts.txt \
  --round 3

python3 ./utils/normalize_text_for_tts.py \
  --input ${CLARIFICATION_DIR}/round_3_user_response.tts.txt \
  --output ${CLARIFICATION_DIR}/round_3_user_response.tts.txt.norm

# Recognize the audio generated from the TTS
sbatch ./utils/decode_response.sh ${TTS_DIR}/instruction_round3_gen ${CLARIFICATION_DIR}/round_3_user_response

sort -o ${CLARIFICATION_DIR}/round_3_user_response/recog.txt ${CLARIFICATION_DIR}/round_3_user_response/recog.txt

# Round 3: Final round of clarification
python ./utils/prepare_batch_dialogue.py \
  ${CLARIFICATION_DIR}/round_3_final_asr_fixed.not_finalized.txt \
  -o ${CLARIFICATION_DIR}/round_4_dialogue.batch.jsonl \
  --round 4 \
  --round1-questions ${CLARIFICATION_DIR}/round_3_to_round4_clarification_questions.txt \
  --round1-answers ${CLARIFICATION_DIR}/round_3_user_response/recog.txt \
  --original-asr ${CLARIFICATION_DIR}/round_3_final_asr_fixed.not_finalized.txt

python ./utils/create_batch_infer.py \
  ${CLARIFICATION_DIR}/round_4_dialogue.batch.jsonl \
  ${CLARIFICATION_DIR}/round_4_dialogue_response.json

python ./utils/extract_text_from_dialogue.py \
  ${CLARIFICATION_DIR}/round_4_dialogue_response.json \
  ${CLARIFICATION_DIR}/round_4_to_round5_clarification_questions.txt \
  ${CLARIFICATION_DIR}/round_4_final_asr_fixed.txt

python3 ./utils/split_finalized_asr.py \
  --input ${CLARIFICATION_DIR}/round_4_final_asr_fixed.txt \
  --finalized ${CLARIFICATION_DIR}/round_4_final_asr_fixed.finalized.txt \
  --not-finalized ${CLARIFICATION_DIR}/round_4_final_asr_fixed.not_finalized.txt

# Combine all finalized ASR from all rounds and clean up tags
cat ${CLARIFICATION_DIR}/round_4_final_asr_fixed.txt \
    ${CLARIFICATION_DIR}/round_3_final_asr_fixed.finalized.txt \
    ${CLARIFICATION_DIR}/round_2_final_asr_fixed.finalized.txt \
    ${CLARIFICATION_DIR}/round_1_final_asr_fixed.finalized.txt | \
    sed 's/<FINAL>//g' | \
    sed 's/<unknown>//g' | \
    sed 's/<del>//g' | \
    sed 's/<unclear>//g' | \
    sed 's/<\/unknown>//g' | \
    sed 's/<\/unclear>//g' > \
    ${CLARIFICATION_DIR}/round_4_final_asr_fixed.finalized.txt

sort -o ${CLARIFICATION_DIR}/round_4_final_asr_fixed.finalized.txt ${CLARIFICATION_DIR}/round_4_final_asr_fixed.finalized.txt