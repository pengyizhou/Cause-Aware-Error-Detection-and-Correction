#!/usr/bin/env python3
"""
run_evaluation.py

This script evaluates model answers against reference answers using GPT-5.2 as a judge.

It uses the eval_gpt.py module to score the answers.

Usage:
    python run_evaluation.py \
        --text data/alpaca_audio_test/text \
        --reference data/alpaca_audio_test/answer.txt \
        --prediction data/alpaca_audio_test/result.txt \
        --output data/alpaca_audio_test/evaluation_results.json \
        --mode binary  # or 'scale' for 0-5 scale scoring
"""

import os
import json
import argparse
from eval_gpt import gpt5_as_judge, gpt5_as_judge_binary


def parse_kaldi_format(file_path):
    """Parse a Kaldi-style file (uttid content) and return a dict."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', 1)
            if len(parts) == 2:
                uttid, content = parts
                data[uttid] = content
            elif len(parts) == 1:
                # Handle case with only uttid (empty content)
                data[parts[0]] = ""
    return data


def load_data(text_path, reference_path, prediction_path):
    """Load and align questions, references, and predictions."""
    
    questions_dict = parse_kaldi_format(text_path)
    references_dict = parse_kaldi_format(reference_path)
    predictions_dict = parse_kaldi_format(prediction_path)
    
    # Align data by uttid (use questions as the primary key)
    questions = []
    references = []
    predictions = []
    uttids = []
    
    for uttid in questions_dict:
        if uttid in references_dict and uttid in predictions_dict:
            uttids.append(uttid)
            questions.append(questions_dict[uttid])
            references.append(references_dict[uttid])
            predictions.append(predictions_dict[uttid])
        else:
            print(f"Warning: uttid {uttid} missing in reference or prediction, skipping.")
    
    print(f"Loaded {len(questions)} aligned samples")
    return uttids, questions, references, predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate model answers using GPT-5.2 as judge")
    parser.add_argument("--text", type=str, required=True,
                        help="Path to the text file containing questions")
    parser.add_argument("--reference", type=str, required=True,
                        help="Path to the reference answers (answer.txt)")
    parser.add_argument("--prediction", type=str, required=True,
                        help="Path to the model predictions (result.txt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save evaluation results (JSON)")
    parser.add_argument("--mode", type=str, default="binary", choices=["binary", "scale"],
                        help="Evaluation mode: 'binary' (0/1) or 'scale' (0-5)")
    
    args = parser.parse_args()
    
    # Load data
    uttids, questions, references, predictions = load_data(
        args.text, args.reference, args.prediction
    )
    
    # Prepare input data for evaluation
    input_data = (questions, references, predictions)
    
    # Run evaluation
    print(f"Running evaluation in '{args.mode}' mode...")
    if args.mode == "binary":
        judge_results, all_details = gpt5_as_judge_binary(None, input_data)
    else:
        judge_results, all_details = gpt5_as_judge(None, input_data)
    
    # Add uttids to details
    for i, detail in enumerate(all_details):
        if i < len(uttids):
            detail['uttid'] = uttids[i]
    
    # Prepare output
    output_data = {
        'summary': judge_results,
        'details': all_details
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Judge Score: {judge_results['judge_score']:.2f}")
    print(f"Success Rate: {judge_results['success_rate']:.2%}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()

