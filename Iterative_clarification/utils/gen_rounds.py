#!/usr/bin/env python3

import sys
import os
import random
import argparse
from pathlib import Path

# Add third-party path to sys.path (can be overridden via THIRD_PARTY_PATH env var)
third_party_path = os.getenv('THIRD_PARTY_PATH', '<PATH_TO_THIRD_PARTY>')
sys.path.append(f'{third_party_path}/Matcha-TTS')

from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio


def load_reference_pairs(ref_folder, max_words=10):
    """Load all reference audio/text pairs from the folder and cache them.
    Only includes references with text length <= max_words.
    """
    ref_folder = Path(ref_folder)
    ref_pairs = []
    skipped_count = 0
    
    # Find all .txt files and match them with corresponding .wav files
    txt_files = sorted(ref_folder.glob('*.txt'))
    for txt_file in txt_files:
        wav_file = txt_file.with_suffix('.wav')
        if wav_file.exists():
            # Read the text content
            with open(txt_file, 'r', encoding='utf-8') as f:
                ref_text = f.read().strip()
            
            # Count words and filter
            word_count = len(ref_text.split())
            if word_count <= max_words:
                ref_pairs.append({
                    'text': ref_text,
                    'wav_path': str(wav_file)
                })
            else:
                skipped_count += 1
    
    print(f"Loaded {len(ref_pairs)} reference pairs from {ref_folder} (skipped {skipped_count} with >{max_words} words)")
    return ref_pairs


def parse_tts_input(input_file):
    """Parse the TTS input file with format: uttid text"""
    targets = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first space to get uttid and text
            parts = line.split(' ', 1)
            if len(parts) == 2:
                uttid, text = parts
                targets.append({'uttid': uttid, 'text': text})
            else:
                print(f"Warning: Skipping malformed line: {line}")
    return targets


def gen_round1(cosyvoice, ref_pairs, targets, output_dir):
    """Generate TTS for all targets using randomly selected reference pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(targets)
    for idx, target in enumerate(targets, 1):
        uttid = target['uttid']
        target_text = target['text']
        
        # Randomly pick a reference pair
        ref_pair = random.choice(ref_pairs)
        ref_text = ref_pair['text']
        ref_wav = ref_pair['wav_path']
        
        print(f"[{idx}/{total}] Processing {uttid}...")
        print(f"  Reference: {Path(ref_wav).name}")
        print(f"  Target text: {target_text[:50]}...")
        
        # Generate TTS
        output_path = output_dir / f"{uttid}.wav"
        try:
            for i, j in enumerate(cosyvoice.inference_zero_shot(target_text, "You are a helpful assistant.<|endofprompt|>" + ref_text, ref_wav, stream=False)):
                torchaudio.save(str(output_path), j['tts_speech'], cosyvoice.sample_rate)
            print(f"  Saved to {output_path}")
        except Exception as e:
            print(f"  Error processing {uttid}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Generate TTS audio using CosyVoice zero-shot synthesis with reference audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 gen_round1.py \\
    --ref_folder <PATH_TO_REFERENCE_AUDIO> \\
    --tts_input_file <PATH_TO_TTS_INPUT> \\
    --output_dir <PATH_TO_OUTPUT> \\
    --model_dir <PATH_TO_MODEL> \\
    --max_words 10
"""
    )
    parser.add_argument(
        '--ref_folder',
        type=str,
        required=True,
        help='Path to folder containing reference audio/text pairs (.wav and .txt files)'
    )
    parser.add_argument(
        '--tts_input_file',
        type=str,
        required=True,
        help='Path to TTS input file (format: uttid text)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Path to output directory for generated audio files'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='<PATH_TO_PRETRAINED_MODELS>/Cosyvoice3-0.5B',
        help='Path to CosyVoice model directory (default: <PATH_TO_PRETRAINED_MODELS>/Cosyvoice3-0.5B)'
    )
    parser.add_argument(
        '--max_words',
        type=int,
        default=10,
        help='Maximum number of words in reference text (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading CosyVoice model...")
    cosyvoice = AutoModel(model_dir=args.model_dir)
    
    # Cache reference pairs
    print("Loading reference pairs...")
    ref_pairs = load_reference_pairs(args.ref_folder, max_words=args.max_words)
    if not ref_pairs:
        print(f"Error: No reference pairs found in {args.ref_folder}")
        return
    
    # Parse TTS input file
    print(f"Reading TTS input file: {args.tts_input_file}")
    targets = parse_tts_input(args.tts_input_file)
    if not targets:
        print(f"Error: No targets found in {args.tts_input_file}")
        return
    
    print(f"Found {len(targets)} targets to process")
    
    # Generate TTS
    gen_round1(cosyvoice, ref_pairs, targets, args.output_dir)
    
    print("Done!")


if __name__ == '__main__':
    main()