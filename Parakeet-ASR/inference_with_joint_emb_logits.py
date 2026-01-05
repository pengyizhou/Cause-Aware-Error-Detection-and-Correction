#!/home/asrxiv/home2/anaconda3/envs/icefall-nemo/bin/python3.10

import nemo.collections.asr as nemo_asr
import torch
import ipdb
import os
import argparse
import sys
import json
from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig

from nemo.collections.asr.parts.utils.asr_confidence_utils import (
    ConfidenceConfig,
    ConfidenceConstants,
    ConfidenceMethodConfig,
    ConfidenceMethodConstants,
)

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
confidence_cfg = ConfidenceConfig(
    preserve_frame_confidence=False, # Internally set to true if preserve_token_confidence == True
    # or preserve_word_confidence == True
    preserve_token_confidence=False, # Internally set to true if preserve_word_confidence == True
    preserve_word_confidence=True,
    aggregation="prod", # How to aggregate frame scores to token scores and token scores to word scores
    exclude_blank=False, # If true, only non-blank emissions contribute to confidence scores
    tdt_include_duration=False, # If true, calculate duration confidence for the TDT models
    method_cfg=ConfidenceMethodConfig( # Config for per-frame scores calculation (before aggregation)
        name="entropy",
        entropy_type="tsallis",
        alpha=0.33,  # Parameter for entropy calculation
        entropy_norm="exp",  # Normalization method
    )
)
asr_model.change_decoding_strategy(
    RNNTDecodingConfig(confidence_cfg=confidence_cfg)
)
def transcribe_with_joint_embeddings(audio_files, audio_ids, result_dir):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    embed_output_path = os.path.join(result_dir, "joint_embeddings.pt")
    logits_output_path = os.path.join(result_dir, "logits.pt")
    transcription_path = os.path.join(result_dir, "recog.txt")
    timestamp_path = os.path.join(result_dir, "timestamps.txt")
    token_timestamp_path = os.path.join(result_dir, "token_timestamps.txt")
    confidence_path = os.path.join(result_dir, "confidence.json")  # JSON format for structured data
    
    all_embeddings = {}  # Dict to store embeddings: audio_id -> tensor of shape [time, 640]
    all_logits = {}  # Dict to store logits: audio_id -> tensor of shape [time, 1030]
    all_confidence = {}  # Dict to store confidence: audio_id -> dict with word_confidence, token_confidence, etc.
    
    # Process files one at a time with batch_size=1
    successful_files = []
    failed_files = []
    with open(token_timestamp_path, "w") as f_token_ts:
        with open(transcription_path, "w") as f:
            with open(timestamp_path, "w") as f_ts:
                for idx, (audio_file, audio_id) in enumerate(zip(audio_files, audio_ids)):
                    try:
                        print(f"Processing {audio_id}")
                        # Confidence configuration is already set on the decoder
                        output = asr_model.transcribe(
                            [audio_file], 
                            timestamps=True, 
                            batch_size=1
                        )
                        result = output[0]
                        f.write(f"{audio_id}\t{result.text}\n")
                        # Extract timestamps
                        frame_predictions = result.timestamp['word']
                        char_predictions = result.timestamp['char']
                        f_token_ts.write(f"{audio_id}\t{str(char_predictions)}\n")
                        # Write word-level timestamps
                        f_ts.write(f"{audio_id}\t{str(frame_predictions)}\n")
                        
                        # Store confidence in structured format
                        confidence_data = {
                            'word_confidence': result.word_confidence if hasattr(result, 'word_confidence') and result.word_confidence is not None else None,
                        }
                        all_confidence[audio_id] = confidence_data
                        
                        # Extract joint embeddings from alignments
                        if result.alignments is not None and len(result.alignments) > 2:
                            embeddings_list = []
                            logits_list = []
                            for align_entry in result.alignments:
                                if len(align_entry) > 0 and len(align_entry[0]) > 2:
                                    if align_entry[0][1] != 1024: # 1024 is the token for padding
                                        logits = align_entry[0][0]  # Get the logits (1030-dim)
                                        logits_list.append(logits)
                                        emb = align_entry[0][2]  # Get the joint embedding (640-dim)
                                        embeddings_list.append(emb)
                            
                            if embeddings_list:
                                all_embeddings[audio_id] = torch.stack(embeddings_list, dim=0)
                            if logits_list:
                                all_logits[audio_id] = torch.stack(logits_list, dim=0)
                        successful_files.append(audio_id)
                    except (IndexError, Exception) as e:
                        error_msg = f"Error processing {audio_id}: {str(e)}"
                        print(f"ERROR: {error_msg}", file=sys.stderr)
                        failed_files.append((audio_id, error_msg))
    
    if failed_files:
        print(f"\nWarning: {len(failed_files)} file(s) failed to process:", file=sys.stderr)
        for audio_id, error in failed_files:
            print(f"  - {audio_id}: {error}", file=sys.stderr)
    
    print(f"\nSuccessfully processed {len(successful_files)}/{len(audio_files)} files.")
    
    # Save all embeddings to a single .pt file
    torch.save(all_embeddings, embed_output_path)
    print(f"Saved joint embeddings for {len(all_embeddings)} files to {embed_output_path}")
    
    # Save all logits to a single .pt file
    torch.save(all_logits, logits_output_path)
    print(f"Saved logits for {len(all_logits)} files to {logits_output_path}")
    
    # Save all confidence scores to a JSON file
    # Convert numpy/torch arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        """Convert numpy arrays and torch tensors to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)  # Fallback for other types
    
    confidence_serializable = {audio_id: convert_to_serializable(conf_data) 
                               for audio_id, conf_data in all_confidence.items()}
    
    with open(confidence_path, 'w') as f:
        json.dump(confidence_serializable, f, indent=2)
    print(f"Saved confidence scores for {len(all_confidence)} files to {confidence_path}")

def parse_audio_files(audio_dir: str = "path/to/your/audio/files"):

    audio_files = []
    audio_ids = []
    for file_name in os.listdir(audio_dir):
        if file_name.endswith('.wav'):
            audio_files.append(os.path.join(audio_dir, file_name))
            audio_ids.append(os.path.basename(file_name).replace('.wav', ''))
    return audio_files, audio_ids

def parse_audio_wav_scp(audio_scp_path: str):
    audio_files = []
    audio_ids = []
    with open(audio_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                audio_files.append(parts[1])
                audio_ids.append(parts[0])
    return audio_files, audio_ids


def main():
    
    arg_parser = argparse.ArgumentParser(description="Transcribe audio files with timestamps using Parakeet ASR model")
    arg_parser.add_argument(
        '--audio_dir',
        type=str,
        required=False,
        help='Directory containing audio files for transcription')
    arg_parser.add_argument("--audio_scp", type=str, required=False, help="Path to audio wav.scp file")
    args = arg_parser.add_argument("--result_dir", type=str, required=True, help="Directory to save transcription results")
    args = arg_parser.parse_args()
    
    if args.audio_dir:
        audio_files, audio_ids = parse_audio_files(args.audio_dir)
    elif args.audio_scp:
        audio_files, audio_ids = parse_audio_wav_scp(args.audio_scp)
    else:
        raise ValueError("Either --audio_dir or --audio_scp must be provided.")
    print(f"Found {len(audio_files)} audio files for transcription.")
    transcribe_with_joint_embeddings(audio_files, audio_ids, args.result_dir)

if __name__ == "__main__":
    main()