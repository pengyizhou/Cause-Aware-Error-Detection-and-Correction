#!/usr/bin/env python3
"""
Evaluate recall for misrecognized words using normalized confidence scores.

Given normalized confidence scores and ground truth labels, this script:
1. Converts confidence scores to error probabilities (1 - confidence)
2. Finds the threshold that achieves a target False Positive Rate (FPR)
3. Calculates recall for misrecognized words at that threshold

Can either use pre-computed labels or compute labels from hyp/ref alignment.

Usage:
    # Using pre-computed labels:
    python evaluate_confidence_recall.py \
        --confidence_json <path_to_confidence.json> \
        --labels_file <path_to_labels.txt> \
        --target_fpr <target_fpr_value>
    
    # Using hyp/ref alignment:
    python evaluate_confidence_recall.py \
        --confidence_json <path_to_confidence.json> \
        --hyp_file <path_to_hyp.txt> \
        --ref_file <path_to_ref.txt> \
        --target_fpr <target_fpr_value>
"""

import json
import sys
import argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from typing import List, Tuple, Dict


def parse_transcription_file(trans_file: str) -> Dict[str, List[str]]:
    """
    Parse transcription file with format: audio_id transcription_text
    
    Args:
        trans_file: Path to transcription file
    
    Returns:
        Dictionary mapping audio_id to list of words
    """
    trans_dict = {}
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by whitespace - first token is audio_id, rest is transcription
            parts = line.split(None, 1)
            if len(parts) < 2:
                # Empty transcription
                audio_id = parts[0] if parts else ""
                trans_dict[audio_id] = []
            else:
                audio_id = parts[0]
                transcription = parts[1]
                # Split transcription into words
                words = transcription.split()
                trans_dict[audio_id] = words
    
    return trans_dict


def align_hyp_ref(hyp_words: List[str], ref_words: List[str]) -> Tuple[List[int], List[str]]:
    """
    Align hypothesis and reference using edit distance (Levenshtein distance).
    
    Returns alignment labels for hypothesis words:
    - 0 = correct (match)
    - 1 = substitution error
    - 1 = insertion error (hyp has word not in ref)
    
    Note: Deletions in hyp (ref has word not in hyp) don't create labels for hyp,
    but we track them for statistics.
    
    Args:
        hyp_words: List of hypothesis words
        ref_words: List of reference words
    
    Returns:
        Tuple of (labels, alignment_info)
        - labels: List of labels (0 or 1) for each hyp word
        - alignment_info: List of alignment operations ('match', 'sub', 'ins', 'del')
    """
    m, n = len(hyp_words), len(ref_words)
    
    # Dynamic programming table for edit distance
    # dp[i][j] = minimum edit distance between hyp[:i] and ref[:j]
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # i insertions
    for j in range(n + 1):
        dp[0][j] = j  # j deletions
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_words[i-1].lower() == ref_words[j-1].lower():
                # Match (case-insensitive)
                dp[i][j] = dp[i-1][j-1]
            else:
                # Choose minimum of substitution, insertion, deletion
                dp[i][j] = min(
                    dp[i-1][j-1] + 1,  # substitution
                    dp[i-1][j] + 1,    # insertion (hyp has extra word)
                    dp[i][j-1] + 1     # deletion (ref has extra word)
                )
    
    # Backtrack to get alignment
    labels = []
    alignment_info = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and hyp_words[i-1].lower() == ref_words[j-1].lower():
            # Match
            labels.insert(0, 0)
            alignment_info.insert(0, 'match')
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            labels.insert(0, 1)
            alignment_info.insert(0, 'sub')
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            # Insertion (hyp has extra word)
            labels.insert(0, 1)
            alignment_info.insert(0, 'ins')
            i -= 1
        else:
            # Deletion (ref has extra word, skip in hyp)
            alignment_info.insert(0, 'del')
            j -= 1
    
    return labels, alignment_info


def create_labels_from_alignment(
    hyp_dict: Dict[str, List[str]], 
    ref_dict: Dict[str, List[str]],
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Create labels from hypothesis and reference transcriptions using edit distance alignment.
    
    Args:
        hyp_dict: Dictionary mapping audio_id to list of hypothesis words
        ref_dict: Dictionary mapping audio_id to list of reference words
        verbose: Whether to print statistics
    
    Returns:
        Dictionary mapping audio_id to numpy array of labels (0 = correct, 1 = error)
    """
    labels_dict = {}
    stats = {'total': 0, 'matched': 0, 'missing_hyp': 0, 'missing_ref': 0}
    alignment_stats = {'match': 0, 'sub': 0, 'ins': 0, 'del': 0}
    
    all_audio_ids = set(hyp_dict.keys()) | set(ref_dict.keys())
    
    for audio_id in sorted(all_audio_ids):
        stats['total'] += 1
        
        if audio_id not in hyp_dict:
            stats['missing_hyp'] += 1
            if verbose:
                print(f"Warning: Hypothesis not found for {audio_id}", file=sys.stderr)
            continue
        
        if audio_id not in ref_dict:
            stats['missing_ref'] += 1
            if verbose:
                print(f"Warning: Reference not found for {audio_id}", file=sys.stderr)
            continue
        
        hyp_words = hyp_dict[audio_id]
        ref_words = ref_dict[audio_id]
        
        # Align and get labels
        labels, alignment_info = align_hyp_ref(hyp_words, ref_words)
        
        # Update statistics
        for op in alignment_info:
            if op in alignment_stats:
                alignment_stats[op] += 1
        
        labels_dict[audio_id] = np.array(labels, dtype=np.int32)
        stats['matched'] += 1
    
    if verbose:
        print(f"\nAlignment Statistics:", file=sys.stderr)
        print(f"  Total audio IDs: {stats['total']}", file=sys.stderr)
        print(f"  Matched: {stats['matched']}", file=sys.stderr)
        print(f"  Missing hypothesis: {stats['missing_hyp']}", file=sys.stderr)
        print(f"  Missing reference: {stats['missing_ref']}", file=sys.stderr)
        print(f"\nAlignment Operations:", file=sys.stderr)
        print(f"  Matches: {alignment_stats['match']}", file=sys.stderr)
        print(f"  Substitutions: {alignment_stats['sub']}", file=sys.stderr)
        print(f"  Insertions: {alignment_stats['ins']}", file=sys.stderr)
        print(f"  Deletions: {alignment_stats['del']}", file=sys.stderr)
        total_ops = sum(alignment_stats.values())
        if total_ops > 0:
            error_rate = (alignment_stats['sub'] + alignment_stats['ins'] + alignment_stats['del']) / total_ops
            print(f"  Error Rate: {error_rate:.4f} ({error_rate*100:.2f}%)", file=sys.stderr)
    
    return labels_dict


def parse_labels_file(labels_path):
    """
    Parse labels file with format: audio_id\t[token_index:label, token_index:label, ...]
    
    Args:
        labels_path: Path to labels file
    
    Returns:
        Dictionary mapping audio_id to numpy array of labels (0 = correct, 1 = error)
    """
    labels_dict = {}
    with open(labels_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab to get audio_id and label list
            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"Warning: Skipping malformed line: {line}", file=sys.stderr)
                continue
            
            audio_id = parts[0]
            label_str = parts[1]
            
            # Parse label list: [token_index:label, token_index:label, ...]
            parsed_labels = []
            # Remove brackets
            label_str = label_str.strip("[]")
            if label_str:
                # Split by comma and extract both token_index and label
                for item in label_str.split(","):
                    item = item.strip()
                    if ":" in item:
                        token_idx_str, label_str_val = item.split(":", 1)
                        token_index = int(token_idx_str.strip())
                        label = int(label_str_val.strip())
                        parsed_labels.append({
                            'token_index': token_index,
                            'label': label
                        })
            
            # Convert to numpy array sorted by token_index
            sorted_labels = sorted(parsed_labels, key=lambda x: x['token_index'])
            labels_array = np.array([item['label'] for item in sorted_labels], dtype=np.int32)
            
            labels_dict[audio_id] = labels_array
    
    return labels_dict


def load_confidence_scores(confidence_json_path):
    """
    Load confidence scores from JSON file.
    
    Args:
        confidence_json_path: Path to confidence.json file
    
    Returns:
        Dictionary mapping audio_id to numpy array of confidence scores
    """
    with open(confidence_json_path, "r") as f:
        data = json.load(f)
    
    confidence_dict = {}
    for audio_id, entry in data.items():
        if "word_confidence" in entry:
            confidence_dict[audio_id] = np.array(entry["word_confidence"], dtype=np.float32)
        else:
            print(f"Warning: No 'word_confidence' found for {audio_id}", file=sys.stderr)
    
    return confidence_dict


def find_threshold_for_fpr(p_error, labels, target_fpr):
    """
    Find the threshold that achieves a target False Positive Rate (FPR).
    
    FPR = (number of correct tokens predicted as error) / (total correct tokens)
    
    Args:
        p_error: Array of shape [N] - error probabilities for entire dataset
        labels: Array of shape [N] - ground truth labels (1 = error, 0 = correct)
        target_fpr: Desired False Positive Rate (e.g., 0.03 for 3%)
    
    Returns:
        threshold: Probability threshold that achieves target FPR
        actual_fpr: Actual FPR achieved with this threshold
    """
    # Get probabilities for correct tokens only (label=0)
    correct_mask = (labels == 0)
    p_correct = p_error[correct_mask]
    
    if len(p_correct) == 0:
        raise ValueError("No correct tokens found in labels!")
    
    # Sort probabilities in descending order
    # Higher probabilities = more likely to be predicted as error
    sorted_probs = np.sort(p_correct)[::-1]
    
    # Calculate how many correct tokens should be predicted as errors
    # to achieve target FPR
    num_correct = len(p_correct)
    num_false_positives = int(np.ceil(target_fpr * num_correct))
    
    if num_false_positives == 0:
        # If target FPR is very low, use the highest probability
        threshold = sorted_probs[0] if len(sorted_probs) > 0 else 1.0
    elif num_false_positives >= num_correct:
        # If target FPR is very high, use the lowest probability
        threshold = sorted_probs[-1] if len(sorted_probs) > 0 else 0.0
    else:
        # Use the probability at the position that gives us target FPR
        threshold = sorted_probs[num_false_positives - 1]
    
    # Calculate actual FPR with this threshold
    pred_correct = (p_correct >= threshold).astype(int)
    actual_fpr = pred_correct.sum() / num_correct
    
    return threshold, actual_fpr


def evaluate_recall_at_fpr(confidence_dict, labels_dict, target_fpr, verbose=True):
    """
    Evaluate recall for misrecognized words at a target FPR.
    
    Args:
        confidence_dict: Dictionary mapping audio_id to confidence scores array
        labels_dict: Dictionary mapping audio_id to labels array (0 = correct, 1 = error)
        target_fpr: Target False Positive Rate (e.g., 0.03 for 3%)
        verbose: Whether to print detailed information
    
    Returns:
        Dictionary with evaluation results
    """
    # Match and concatenate confidence scores and labels
    all_confidence = []
    all_labels = []
    matched_count = 0
    missing_confidence_count = 0
    missing_labels_count = 0
    length_mismatch_count = 0
    
    # Get all unique audio IDs
    all_audio_ids = set(confidence_dict.keys()) | set(labels_dict.keys())
    
    for audio_id in sorted(all_audio_ids):
        if audio_id not in confidence_dict:
            missing_confidence_count += 1
            if verbose:
                print(f"Warning: Confidence not found for {audio_id}", file=sys.stderr)
            continue
        
        if audio_id not in labels_dict:
            missing_labels_count += 1
            if verbose:
                print(f"Warning: Labels not found for {audio_id}", file=sys.stderr)
            continue
        
        conf_scores = confidence_dict[audio_id]
        labels = labels_dict[audio_id]
        
        # Check length match
        if len(conf_scores) != len(labels):
            length_mismatch_count += 1
            if verbose:
                print(f"Warning: Length mismatch for {audio_id}: "
                      f"confidence={len(conf_scores)}, labels={len(labels)}", file=sys.stderr)
            continue
        
        all_confidence.append(conf_scores)
        all_labels.append(labels)
        matched_count += 1
    
    if len(all_confidence) == 0:
        raise ValueError("No matching confidence scores and labels found!")
    
    # Concatenate all arrays
    all_confidence = np.concatenate(all_confidence)
    all_labels = np.concatenate(all_labels)
    
    if verbose:
        print(f"\nMatched {matched_count} files", file=sys.stderr)
        print(f"Total tokens: {len(all_confidence)}", file=sys.stderr)
        print(f"Correct tokens (label=0): {(all_labels == 0).sum()}", file=sys.stderr)
        print(f"Error tokens (label=1): {(all_labels == 1).sum()}", file=sys.stderr)
        if missing_confidence_count > 0:
            print(f"Missing confidence: {missing_confidence_count} files", file=sys.stderr)
        if missing_labels_count > 0:
            print(f"Missing labels: {missing_labels_count} files", file=sys.stderr)
        if length_mismatch_count > 0:
            print(f"Length mismatches: {length_mismatch_count} files", file=sys.stderr)
    
    # Convert confidence scores to error probabilities
    # Lower confidence = higher error probability
    p_error = 1.0 - all_confidence
    
    # Find threshold for target FPR
    threshold, actual_fpr = find_threshold_for_fpr(p_error, all_labels, target_fpr)
    
    # Make predictions: predict error if p_error >= threshold
    predictions = (p_error >= threshold).astype(int)
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, predictions, average="binary", zero_division=0
    )
    
    # Calculate additional metrics
    true_positives = ((predictions == 1) & (all_labels == 1)).sum()
    false_positives = ((predictions == 1) & (all_labels == 0)).sum()
    false_negatives = ((predictions == 0) & (all_labels == 1)).sum()
    true_negatives = ((predictions == 0) & (all_labels == 0)).sum()
    
    result = {
        "target_fpr": target_fpr,
        "actual_fpr": actual_fpr,
        "threshold": float(threshold),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "true_negatives": int(true_negatives),
        "total_tokens": int(len(all_labels)),
        "error_tokens": int((all_labels == 1).sum()),
        "correct_tokens": int((all_labels == 0).sum()),
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate recall for misrecognized words using confidence scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using pre-computed labels:
  python evaluate_confidence_recall.py \\
      --confidence_json confidence.json \\
      --labels_file labels.txt \\
      --target_fpr 0.03
  
  # Using hyp/ref alignment:
  python evaluate_confidence_recall.py \\
      --confidence_json confidence.json \\
      --hyp_file hypothesis.txt \\
      --ref_file reference.txt \\
      --target_fpr 0.03
        """
    )
    parser.add_argument(
        "--confidence_json",
        type=str,
        required=True,
        help="Path to confidence.json file with normalized confidence scores"
    )
    
    # Either labels_file OR (hyp_file + ref_file) must be provided
    label_group = parser.add_mutually_exclusive_group(required=True)
    label_group.add_argument(
        "--labels_file",
        type=str,
        help="Path to labels file (format: audio_id\\t[token_index:label, ...])"
    )
    label_group.add_argument(
        "--hyp_file",
        type=str,
        help="Path to hypothesis transcription file (format: audio_id transcription)"
    )
    
    parser.add_argument(
        "--ref_file",
        type=str,
        help="Path to reference transcription file (format: audio_id transcription). "
             "Required if --hyp_file is provided."
    )
    parser.add_argument(
        "--target_fpr",
        type=float,
        required=True,
        help="Target False Positive Rate (e.g., 0.03 for 3%%)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.hyp_file and not args.ref_file:
        parser.error("--ref_file is required when --hyp_file is provided")
    
    verbose = not args.quiet
    
    # Load confidence scores
    if verbose:
        print(f"Loading confidence scores from {args.confidence_json}...", file=sys.stderr)
    confidence_dict = load_confidence_scores(args.confidence_json)
    
    # Load or create labels
    if args.labels_file:
        if verbose:
            print(f"Loading labels from {args.labels_file}...", file=sys.stderr)
        labels_dict = parse_labels_file(args.labels_file)
    else:
        if verbose:
            print(f"Loading hypothesis from {args.hyp_file}...", file=sys.stderr)
        hyp_dict = parse_transcription_file(args.hyp_file)
        
        if verbose:
            print(f"Loading reference from {args.ref_file}...", file=sys.stderr)
        ref_dict = parse_transcription_file(args.ref_file)
        
        if verbose:
            print(f"Performing edit distance alignment...", file=sys.stderr)
        labels_dict = create_labels_from_alignment(hyp_dict, ref_dict, verbose=verbose)
    
    # Evaluate
    if verbose:
        print(f"\nEvaluating at target FPR = {args.target_fpr}...", file=sys.stderr)
    
    result = evaluate_recall_at_fpr(
        confidence_dict,
        labels_dict,
        args.target_fpr,
        verbose=verbose
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Target FPR:        {result['target_fpr']:.4f}")
    print(f"Actual FPR:        {result['actual_fpr']:.4f}")
    print(f"Threshold:         {result['threshold']:.6f}")
    print(f"\nRecall:            {result['recall']:.4f} ({result['recall']*100:.2f}%%)")
    print(f"Precision:         {result['precision']:.4f} ({result['precision']*100:.2f}%%)")
    print(f"F1 Score:          {result['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives:  {result['true_positives']}")
    print(f"  False Positives: {result['false_positives']}")
    print(f"  False Negatives: {result['false_negatives']}")
    print(f"  True Negatives:  {result['true_negatives']}")
    print(f"\nTotal Statistics:")
    print(f"  Total Tokens:    {result['total_tokens']}")
    print(f"  Error Tokens:    {result['error_tokens']}")
    print(f"  Correct Tokens:  {result['correct_tokens']}")
    print("="*60)


if __name__ == "__main__":
    main()

