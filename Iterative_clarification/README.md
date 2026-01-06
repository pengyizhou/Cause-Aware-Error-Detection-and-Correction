# Iterative Clarification Pipeline

This directory contains the implementation for a 3-round iterative clarification system that improves ASR accuracy through dialogue-based error correction. The system uses LLM-based clarification questions, TTS generation, and ASR to iteratively refine transcription results.

## Overview

The iterative clarification pipeline addresses ASR errors by:
1. **Initial Round**: Identifying errors in ASR output using error detection models
2. **Clarification Rounds (1-3)**: Generating clarification questions for uncertain segments, synthesizing user responses via TTS, and re-recognizing to improve accuracy
3. **Evaluation**: Assessing the final results on speech-instruction tasks

## Directory Structure

```
Iterative_clarification/
├── README.md                          # This file
├── run_clarification.sh              # Main script for 3-round iterative clarification
├── run_tts.sh                        # TTS generation script (used during each round)
├── run_eval_MaJ.sh                   # Evaluation script for speech-instruction task
└── utils/                            # Utility scripts
    ├── prepare_batch_dialogue.py     # Prepare dialogue batches for LLM
    ├── create_batch_infer.py         # Create batch inference jobs
    ├── extract_text_from_dialogue.py # Extract clarification questions and fixed ASR
    ├── split_finalized_asr.py        # Split ASR into finalized/not-finalized
    ├── prepare_batch_user_response.py # Prepare user response batches
    ├── extract_user_response_direct.py # Extract user response text
    ├── normalize_text_for_tts.py    # Normalize text for TTS input
    ├── gen_rounds.py                  # TTS generation using CosyVoice
    ├── decode_response.sh            # ASR decoding for TTS-generated audio
    ├── dialogue_gpt.py               # Generate dialogue responses from ASR
    ├── eval_gpt.py                   # GPT-based evaluation
    └── run_evaluation.py             # Evaluation metrics computation
```

## Workflow

### 1. Initial Round (Error Detection)

The initial round processes ASR output with error markers:
- Combines ASR results from multiple datasets with error markers
- Generates clarification questions for uncertain segments
- Splits results into finalized (confident) and not-finalized (uncertain) utterances

### 2. Clarification Rounds (1-3)

Each clarification round follows this process:

1. **Dialogue Generation**: Prepare dialogue batches with previous round's questions/answers
2. **LLM Inference**: Generate clarification questions and fixed ASR
3. **ASR Splitting**: Separate finalized and not-finalized utterances
4. **User Response Preparation**: Create user response batches based on clarification questions
5. **TTS Generation**: Synthesize user responses using `run_tts.sh`
6. **ASR Decoding**: Recognize the TTS-generated audio
7. **Iteration**: Continue with remaining not-finalized utterances

### 3. Final Combination

After all rounds, combine all finalized ASR results and clean up special tags.

## Scripts

### `run_clarification.sh`

Main script for the 3-round iterative clarification pipeline.

**Configuration** (update at the top of the script):
- `DATASET1_PREFIX`, `DATASET2_PREFIX`: Dataset name prefixes
- `DATA_DIR`: Path to data directory
- `CLARIFICATION_DIR`: Path to clarification working directory
- `TTS_DIR`: Path to TTS output directory

**Usage**:
```bash
bash run_clarification.sh
```

**Workflow**:
1. **Initial**: Prepare ASR with error markers, generate clarification questions
2. **Round 1**: Process not-finalized utterances from initial round
3. **Round 2**: Process not-finalized utterances from round 1
4. **Round 3**: Process not-finalized utterances from round 2
5. **Final**: Combine all finalized results

**Note**: The script uses `run_tts.sh` (via `sbatch`) during each round to generate TTS audio for user responses.

### `run_tts.sh`

SLURM script for TTS generation using CosyVoice zero-shot synthesis.

**Configuration** (update at the top of the script):
- `REF_FOLDER`: Path to reference audio folder (containing .wav and .txt pairs)
- `TTS_INPUT_FILE`: Path to TTS input file (format: `uttid text`)
- `OUTPUT_DIR`: Path to output directory for generated audio
- `MODEL_DIR`: Path to CosyVoice model directory (optional)
- `MAX_WORDS`: Maximum words in reference text (default: 10)
- `THIRD_PARTY_PATH`: Path to third-party directory containing Matcha-TTS (optional)

**Usage**:
```bash
sbatch run_tts.sh
```

**Details**:
- Uses CosyVoice for zero-shot TTS synthesis
- Randomly selects reference audio/text pairs from the reference folder
- Generates audio files named `{uttid}.wav` in the output directory
- **Note**: For CosyVoice installation and setup, refer to [https://github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)

### `run_eval_MaJ.sh`

Evaluation script for the speech-instruction task using GPT as a judge.

**Configuration** (update at the top of the script):
- `DATASET1`, `DATASET2`: Dataset names
- `ASR_RESULTS_DIR`: Path to ASR results directory
- `DATA_DIR`: Path to data directory

**Usage**:
```bash
bash run_eval_MaJ.sh
```

**Workflow**:
1. Generate dialogue responses from ASR transcriptions using `dialogue_gpt.py`
2. Evaluate responses using GPT-5.2 as a judge with scale scoring (0-5)
3. Output evaluation results as JSON files

## Dependencies

### Python Packages
- `openai`: For LLM API access
- `torchaudio`: For audio processing
- `cosyvoice`: From CosyVoice for TTS synthesis
- Standard libraries: `argparse`, `pathlib`, `json`, `re`

### External Tools
- **CosyVoice**: TTS model (Cosyvoice3-0.5B)
  - For installation and usage instructions, please refer to: [https://github.com/FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
  - The pipeline uses CosyVoice for zero-shot TTS synthesis with reference audio
- **Matcha-TTS**: Third-party TTS framework (included in CosyVoice repository)
- **Parakeet-ASR**: For ASR inference (see `decode_response.sh`)
- **SLURM**: For job scheduling

### Environment Variables
- `THIRD_PARTY_PATH`: Path to third-party directory (for Matcha-TTS, typically within CosyVoice repository)
- `OPENAI_API_KEY`: OpenAI API key (set in environment)

## Input/Output Formats

### TTS Input Format
```
uttid text
```
Example:
```
alpaca_001-round1 Can you clarify what you said?
```

### ASR Output Format
```
uttid transcription
```

### Clarification Questions Format
```
uttid question_text
```

### Finalized/Not-Finalized Format
```
uttid asr_text
```
- Finalized: Contains `<FINAL>` tag or no uncertainty tags
- Not-finalized: Contains uncertainty tags like `<unknown>`, `<del>`, `<unclear>`

## Example Workflow

1. **Setup**: Configure paths in `run_clarification.sh`
2. **Run Initial Round**: Execute `run_clarification.sh` (it will automatically call `run_tts.sh` when needed)
3. **Monitor**: Check logs in `./log/` directory
4. **Evaluate**: After completion, run `run_eval_MaJ.sh` to evaluate results

## Notes

- The pipeline processes utterances iteratively, only continuing with not-finalized segments
- TTS generation uses zero-shot synthesis with randomly selected reference audio
- Each round builds upon previous rounds' questions and answers
- The final output combines all finalized ASR from all rounds
- Special tags (`<FINAL>`, `<unknown>`, `<del>`, `<unclear>`) are removed in the final output

## Troubleshooting

- **TTS Generation Fails**: Check that reference audio folder contains matching .wav and .txt files
- **ASR Decoding Fails**: Verify Parakeet-ASR path in `utils/decode_response.sh`
- **LLM API Errors**: Ensure `OPENAI_API_KEY` is set and API limits are not exceeded
- **Path Issues**: Verify all placeholder paths are updated in configuration sections

