# Cause-Aware Error Detection and Correction

A comprehensive pipeline for detecting and correcting ASR errors through cause-aware classification and iterative clarification.

## Overview

This repository implements a multi-stage system for improving ASR accuracy by:
1. **Error Detection**: Identifying ASR errors and their root causes (distortion types, comprehension issues, perception problems)
2. **Error Classification**: Using frame-level classifiers to detect distortion events and error types
3. **Iterative Clarification**: Employing a 3-round dialogue-based approach to refine uncertain transcriptions

## Repository Structure

### `data_prepare/`
Data preparation and distortion simulation. Generates various types of speech distortions (noise, RIR, interference, packet loss, missing segments) and corresponding frame-level labels.

**See**: [data_prepare/README.md](data_prepare/README.md)

### `Parakeet-ASR/`
ASR inference using NVIDIA Parakeet TDT model. Includes:
- ASR decoding with joint embeddings and logits extraction
- Entropy-based confidence scoring
- Post-ASR label generation pipeline (comprehension, perception, distortion events)

**See**: [Parakeet-ASR/README.md](Parakeet-ASR/README.md)

### `model/`
Frame-level classification models for error detection:
- **Parakeet-encoder-linear**: Detects sound events (6 labels) and deletion errors (2 labels) from encoder features
- **Parakeet-joiner-linear**: Detects comprehension (2 labels) and perception errors (2 labels) from joint embeddings

**See**: [model/README.md](model/README.md)

### `Iterative_clarification/`
3-round iterative clarification pipeline that uses LLM-based dialogue to refine ASR transcriptions through clarification questions and TTS-generated user responses.

**See**: [Iterative_clarification/README.md](Iterative_clarification/README.md)

## Quick Start

1. **Data Preparation**: Generate distorted audio datasets with frame-level labels
   ```bash
   cd data_prepare
   # Follow instructions in README.md
   ```

2. **ASR Inference**: Run ASR on your datasets
   ```bash
   cd Parakeet-ASR
   # Follow instructions in README.md
   ```

3. **Model Training**: Train error detection classifiers
   ```bash
   cd model/Parakeet-encoder-linear  # or Parakeet-joiner-linear
   # Follow instructions in README.md
   ```

4. **Iterative Clarification**: Run the clarification pipeline
   ```bash
   cd Iterative_clarification
   bash run_clarification.sh
   ```

## Dependencies

- **NeMo**: For Parakeet ASR model (see [Parakeet-ASR/README.md](Parakeet-ASR/README.md))
- **CosyVoice**: For TTS generation (see [Iterative_clarification/README.md](Iterative_clarification/README.md))
- **PyTorch**: For model training and inference
- **OpenAI API**: For LLM-based dialogue generation
- **FFmpeg**: For audio processing (configurable via `FFMPEG_PATH` environment variable)

## Citation

If you use this code, please cite the relevant papers and datasets as documented in the subdirectory READMEs.
