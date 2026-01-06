#!/usr/bin/env python3
"""
Script to map classifier predictions to word-level errors and compare with ground truth.

This script:
1. Aligns transcription (ground truth) and recognized text to find word-level errors
2. Maps frame-level classifier predictions to words using token timestamps
3. Computes false-alarm rate and error recall
4. Outputs recognized text with error words marked as <err>...</err>
"""

import json
import ast
import argparse
import os
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re

def load_transcription(transcription_file: str) -> Dict[str, str]:
    """Load transcription file: file_id transcription_text"""
    transcription = {}
    with open(transcription_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by first space or tab
            parts = line.split(None, 1)
            if len(parts) >= 2:
                file_id = parts[0]
                text = parts[1]
                transcription[file_id] = text
    return transcription


def load_recog(recog_file: str) -> Dict[str, str]:
    """Load recognition file: file_id recognized_text"""
    recog = {}
    with open(recog_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by first space or tab
            parts = line.split(None, 1)
            if len(parts) >= 2:
                file_id = parts[0]
                text = parts[1]
                recog[file_id] = text
    return recog


def load_token_timestamps(timestamps_file: str) -> Dict[str, List[Dict]]:
    """Load token timestamps file: file_id [list of token dicts]"""
    timestamps = {}
    with open(timestamps_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split by tab
            parts = line.split('\t', 1)
            if len(parts) >= 2:
                file_id = parts[0]
                tokens_str = parts[1]
                try:
                    # Parse the list of dicts
                    tokens = ast.literal_eval(tokens_str)
                    timestamps[file_id] = tokens
                except:
                    print(f"Warning: Could not parse timestamps for {file_id}")
    return timestamps


def load_classifier_predictions(jsonl_file: str) -> Dict[str, Dict]:
    """Load classifier predictions from JSONL file"""
    predictions = {}
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                file_id = data['file_id']
                preds = data.get('predictions', [])
                predictions[file_id] = {
                    'predictions': preds,
                    'num_frames': data.get('num_frames', len(preds))
                }
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON line: {line[:100]}")
    return predictions


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words, preserving punctuation as separate tokens"""
    # Split by whitespace and keep punctuation attached or separate
    tokens = re.findall(r'\S+', text)
    return tokens


def is_punctuation_only(token: str) -> bool:
    """Check if a token contains only punctuation characters"""
    # Remove all alphanumeric characters and check if anything remains
    cleaned = re.sub(r'[\w\s]', '', token)
    # Also check if the token itself is just punctuation (no alphanumeric)
    has_alnum = bool(re.search(r'[\w]', token))
    return not has_alnum and len(cleaned) > 0


def align_texts(ref_tokens: List[str], hyp_tokens: List[str]) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Align reference and hypothesis texts using edit distance.
    Returns: (substitutions, deletions, insertions)
    Each tuple is (ref_idx, ref_word, hyp_idx, hyp_word, error_type)
    """
    # Use dynamic programming for alignment
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    path = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            path[i][0] = ('del', i-1, 0)
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            path[0][j] = ('ins', 0, j-1)
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            ref_word = ref_tokens[i-1].lower()
            hyp_word = hyp_tokens[j-1].lower()
            
            if ref_word == hyp_word:
                dp[i][j] = dp[i-1][j-1]
                path[i][j] = ('match', i-1, j-1)
            else:
                # Substitution
                sub_cost = dp[i-1][j-1] + 1
                # Deletion
                del_cost = dp[i-1][j] + 1
                # Insertion
                ins_cost = dp[i][j-1] + 1
                
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                
                if min_cost == sub_cost:
                    path[i][j] = ('sub', i-1, j-1)
                elif min_cost == del_cost:
                    path[i][j] = ('del', i-1, j)
                else:
                    path[i][j] = ('ins', i, j-1)
    
    # Backtrack to find alignment
    substitutions = []
    deletions = []
    insertions = []
    i, j = m, n
    
    alignment = []
    while i > 0 or j > 0:
        if path[i][j] is None:
            break
        op, prev_i, prev_j = path[i][j]
        
        if op == 'match':
            alignment.append(('match', i-1, j-1, ref_tokens[i-1], hyp_tokens[j-1]))
            i, j = prev_i, prev_j
        elif op == 'sub':
            substitutions.append((i-1, ref_tokens[i-1], j-1, hyp_tokens[j-1], 'sub'))
            alignment.append(('sub', i-1, j-1, ref_tokens[i-1], hyp_tokens[j-1]))
            i, j = prev_i, prev_j
        elif op == 'del':
            deletions.append((i-1, ref_tokens[i-1], None, None, 'del'))
            alignment.append(('del', i-1, None, ref_tokens[i-1], None))
            i, j = prev_i, prev_j
        elif op == 'ins':
            insertions.append((None, None, j-1, hyp_tokens[j-1], 'ins'))
            alignment.append(('ins', None, j-1, None, hyp_tokens[j-1]))
            i, j = prev_i, prev_j
    
    alignment.reverse()
    return substitutions, deletions, insertions, alignment


def tokens_to_words(tokens: List[Dict]) -> List[Tuple[str, float, float]]:
    """
    Convert token-level timestamps to word-level.
    Returns list of (word, start_time, end_time)
    """
    words = []
    current_word = ""
    word_start = None
    word_end = None
    
    for token in tokens:
        char = token.get('char', [''])[0] if isinstance(token.get('char'), list) else token.get('char', '')
        start = token.get('start', 0.0)
        end = token.get('end', 0.0)
        
        # Check if this token starts a new word (has space before it or is punctuation)
        if word_start is None:
            word_start = start
        
        # Accumulate characters
        current_word += char
        
        # Check if this token ends a word (space or punctuation after)
        # Simple heuristic: if next token has a gap or if current token is punctuation
        word_end = end
        
        # Check if we should end the word
        # Look ahead or use heuristics based on character offsets
        if char.strip() and not char.isalnum() and char not in ["'", "-"]:
            # Punctuation - end current word
            if current_word.strip():
                words.append((current_word.strip(), word_start, word_end))
            current_word = ""
            word_start = None
            word_end = None
        elif 'start_offset' in token:
            # Check if there's a gap indicating word boundary
            # This is a simplified approach - you may need to adjust based on your data
            pass
    
    # Add final word if exists
    if current_word.strip():
        words.append((current_word.strip(), word_start, word_end))
    
    return words


def tokens_to_words_improved(tokens: List[Dict], recog_text: str) -> List[Tuple[str, float, float]]:
    """
    Improved version: Use recognized text to properly segment words.
    Returns list of (word, start_time, end_time)
    """
    # First, extract all characters with their timestamps
    char_timestamps = []
    for token in tokens:
        char = token.get('char', [''])[0] if isinstance(token.get('char'), list) else token.get('char', '')
        start = token.get('start', 0.0)
        end = token.get('end', 0.0)
        char_timestamps.append((char, start, end))
    
    # Reconstruct text from tokens
    token_text = ''.join([char for char, _, _ in char_timestamps])
    
    # Tokenize the recognized text
    recog_words = tokenize_text(recog_text)
    
    # Map words to timestamps by finding their positions in token text
    words = []
    text_pos = 0
    
    for word in recog_words:
        word_clean = word.strip()
        if not word_clean:
            continue
        
        # Find word in token text (case-insensitive search)
        word_lower = word_clean.lower()
        token_text_lower = token_text.lower()
        
        # Search for word starting from current position
        idx = token_text_lower.find(word_lower, text_pos)
        
        if idx != -1:
            # Found word, map to timestamps
            char_count = 0
            word_start_time = None
            word_end_time = None
            
            for i, (char, start, end) in enumerate(char_timestamps):
                char_len = len(char)
                
                # Check if this token overlaps with the word position
                if char_count <= idx < char_count + char_len:
                    if word_start_time is None:
                        word_start_time = start
                    word_end_time = end
                
                # Check if we've covered the entire word
                if char_count + char_len >= idx + len(word_clean):
                    word_end_time = end
                    break
                
                char_count += char_len
            
            if word_start_time is not None:
                words.append((word, word_start_time, word_end_time))
                text_pos = idx + len(word_clean)
            else:
                # Fallback: use first and last token times
                if char_timestamps:
                    words.append((word, char_timestamps[0][1], char_timestamps[-1][2]))
        else:
            # Word not found in token text, use heuristic
            # Estimate based on character count
            if char_timestamps:
                # Use average time
                total_chars = sum(len(char) for char, _, _ in char_timestamps)
                if total_chars > 0:
                    char_ratio = text_pos / total_chars if total_chars > 0 else 0
                    total_time = char_timestamps[-1][2] - char_timestamps[0][1]
                    est_start = char_timestamps[0][1] + char_ratio * total_time
                    est_end = est_start + (len(word_clean) / total_chars) * total_time
                    words.append((word, est_start, est_end))
                    text_pos += len(word_clean)
    
    return words


def map_tokens_to_words(classifier_preds: List[int], tokens: List[Dict], 
                       word_timestamps: List[Tuple[str, float, float]]) -> Dict[int, int]:
    """
    Map token-level predictions to word indices (one-to-one mapping).
    Classifier predictions map one-to-one to tokens.
    If length mismatch, pad classifier_preds at the end with 0.
    Returns dict mapping word_idx -> prediction (0 or 1)
    """
    word_predictions = {}
    
    # Pad classifier_preds if needed (pad with 0 at the end)
    num_tokens = len(tokens)
    if len(classifier_preds) < num_tokens:
        classifier_preds = classifier_preds + [0] * (num_tokens - len(classifier_preds))
    elif len(classifier_preds) > num_tokens:
        classifier_preds = classifier_preds[:num_tokens]
    
    # Build character sequence from tokens to map tokens to words
    token_chars = []
    for token in tokens:
        char = token.get('char', [''])[0] if isinstance(token.get('char'), list) else token.get('char', '')
        token_chars.append(char)
    
    token_text = ''.join(token_chars)
    
    # Map each word to its token range by finding word in token text
    char_pos = 0
    token_idx = 0
    
    for word_idx, (word, start_time, end_time) in enumerate(word_timestamps):
        word_clean = word.strip()
        if not word_clean:
            word_predictions[word_idx] = 0
            continue
        
        # Find word in token text starting from current position
        word_lower = word_clean.lower()
        token_text_lower = token_text.lower()
        
        # Search for word starting from current char position
        word_start_in_text = token_text_lower.find(word_lower, char_pos)
        
        if word_start_in_text != -1:
            # Find which tokens cover this word
            start_token_idx = None
            end_token_idx = None
            current_char_pos = 0
            
            for i, char in enumerate(token_chars):
                char_len = len(char)
                # Check if this token overlaps with word position
                if current_char_pos <= word_start_in_text < current_char_pos + char_len:
                    if start_token_idx is None:
                        start_token_idx = i
                if current_char_pos <= word_start_in_text + len(word_clean) <= current_char_pos + char_len:
                    end_token_idx = i + 1
                    break
                current_char_pos += char_len
            
            if start_token_idx is not None:
                if end_token_idx is None:
                    end_token_idx = start_token_idx + 1
                # Check if any token in this range has error prediction
                # has_error = any(classifier_preds[i] == 1 for i in range(start_token_idx, end_token_idx))
                # Having only majority prediction is wrong, to set it as error
                num_errors = sum(classifier_preds[i] for i in range(start_token_idx, end_token_idx))
                has_error = num_errors > (end_token_idx - start_token_idx) // 2
                # ipdb.set_trace()
                word_predictions[word_idx] = 1 if has_error else 0
                char_pos = word_start_in_text + len(word_clean)
            else:
                word_predictions[word_idx] = 0
        else:
            # Word not found, use simple heuristic: divide tokens evenly
            tokens_per_word = num_tokens / len(word_timestamps) if word_timestamps else 1
            start_token_idx = int(word_idx * tokens_per_word)
            end_token_idx = min(int((word_idx + 1) * tokens_per_word), num_tokens)
            has_error = any(classifier_preds[i] == 1 for i in range(start_token_idx, end_token_idx))
            word_predictions[word_idx] = 1 if has_error else 0
    
    return word_predictions


def compute_metrics(ground_truth_errors: Set[int], predicted_errors: Set[int], 
                   total_words: int) -> Dict[str, float]:
    """
    Compute false-alarm rate and error recall.
    
    False-alarm rate = false positives / (false positives + true negatives)
    Error Recall = true positives / (true positives + false negatives)
    """
    true_positives = len(ground_truth_errors & predicted_errors)
    false_positives = len(predicted_errors - ground_truth_errors)
    false_negatives = len(ground_truth_errors - predicted_errors)
    true_negatives = total_words - len(ground_truth_errors) - false_positives
    
    # False-alarm rate (also called false positive rate)
    false_alarm_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    
    # Error recall (also called sensitivity or true positive rate)
    error_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Precision
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # F1 score
    f1 = 2 * precision * error_recall / (precision + error_recall) if (precision + error_recall) > 0 else 0.0
    
    return {
        'false_alarm_rate': false_alarm_rate,
        'error_recall': error_recall,
        'precision': precision,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives
    }


def detect_deletions(deletion_preds: List[int], tokens: List[Dict], 
                    frame_duration: float = 0.08, tolerance_frames: int = 1) -> List[int]:
    """
    Detect deletion errors based on frame-wise deletion predictions.
    If prediction is 1 and there are no tokens in that frame (with tolerance),
    mark it as a deletion.
    Consecutive deletion frames are grouped together and only one deletion is marked per group.
    
    Args:
        deletion_preds: Frame-wise deletion predictions (0 or 1)
        tokens: List of token dictionaries with 'start' and 'end' times
        frame_duration: Duration of each frame in seconds (default: 0.08)
        tolerance_frames: Number of frames to check around predicted frame (default: 1, so ±1 frame = 3 frames total)
    
    Returns:
        List of frame indices where deletions are detected (one per consecutive group)
    """
    deletion_frames = []
    
    # First pass: collect all frames with deletion predictions and no tokens
    for frame_idx, pred in enumerate(deletion_preds):
        if pred == 1:
            # Calculate frame time range (with tolerance)
            frame_start_time = frame_idx * frame_duration
            frame_end_time = (frame_idx + 1) * frame_duration
            
            # Check tolerance range: -tolerance_frames to +tolerance_frames
            check_start_frame = max(0, frame_idx - tolerance_frames)
            check_end_frame = min(len(deletion_preds), frame_idx + tolerance_frames + 1)
            
            check_start_time = check_start_frame * frame_duration
            check_end_time = check_end_frame * frame_duration
            
            # Check if there are any tokens in this time range
            has_tokens = False
            for token in tokens:
                token_start = token.get('start', 0.0)
                token_end = token.get('end', 0.0)
                
                # Check if token overlaps with the time range
                if not (token_end < check_start_time or token_start > check_end_time):
                    has_tokens = True
                    break
            
            # If no tokens found in the tolerance range, it's a deletion
            if not has_tokens:
                deletion_frames.append(frame_idx)
    
    # Second pass: group consecutive frames and return only one frame per group
    if not deletion_frames:
        return []
    
    grouped_deletions = []
    current_group_start = deletion_frames[0]
    
    for i in range(1, len(deletion_frames)):
        # If frames are consecutive (difference of 1), they're in the same group
        if deletion_frames[i] == deletion_frames[i-1] + 1:
            # Continue the current group
            continue
        else:
            # End of current group, add the first frame of the group
            grouped_deletions.append(current_group_start)
            # Start a new group
            current_group_start = deletion_frames[i]
    
    # Don't forget the last group
    grouped_deletions.append(current_group_start)
    
    return grouped_deletions


def mark_errors_in_text(words: List[str], error_indices: Set[int], 
                        deletion_positions: List[int] = None,
                        error_type_map: Dict[int, str] = None) -> str:
    """
    Mark error words in text using error-specific tags and deletions using <del> tags.
    Words are lowercased but punctuation is preserved.
    
    Args:
        words: List of words (original with punctuation)
        error_indices: Set of word indices that are errors
        deletion_positions: List of word positions for deletions
        error_type_map: Dict mapping word index to error type ('unclear' or 'unknown')
    """
    marked_words = []
    deletion_positions_sorted = sorted(deletion_positions or [])
    deletion_pos_set = set(deletion_positions_sorted)
    error_type_map = error_type_map or {}
    
    for i, word in enumerate(words):
        # Insert deletions before this word if needed
        if i in deletion_pos_set:
            marked_words.append("<del>")
        
        # Lowercase word but keep punctuation
        word_lower = word.lower()
        
        # Mark error words (skip punctuation-only tokens)
        if i in error_indices and not is_punctuation_only(word):
            # Determine error type: 'unclear' for hearing fault, 'unknown' for intelligibility
            error_type = error_type_map.get(i, 'unknown')  # Default to 'unknown'
            if error_type == 'unclear':
                marked_words.append(f"<unclear>{word_lower}</unclear>")
            else:  # 'unknown' or default
                marked_words.append(f"<unknown>{word_lower}</unknown>")
        else:
            marked_words.append(word_lower)
    
    # Add deletions at the end if needed
    for del_pos in deletion_positions_sorted:
        if del_pos >= len(words):
            marked_words.append("<del>")
    
    return " ".join(marked_words)


def main():
    parser = argparse.ArgumentParser(description='Evaluate word-level error predictions')
    parser.add_argument('--transcription', type=str, required=True,
                       help='Path to transcription.txt (ground truth)')
    parser.add_argument('--recog', type=str, required=True,
                       help='Path to recog.txt (ASR output)')
    parser.add_argument('--token-timestamps', type=str, required=True,
                       help='Path to token_timestamps.txt.nooverlap.final.txt')
    parser.add_argument('--hearing-fault-predictions', type=str, default=None,
                       help='Path to hearing fault (cannot_hear_clearly) prediction JSONL file')
    parser.add_argument('--intelligibility-predictions', type=str, default=None,
                       help='Path to intelligibility (cannot_understand) prediction JSONL file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file for marked text')
    parser.add_argument('--output-metrics', type=str, default=None,
                       help='Output file for metrics (JSON format)')
    parser.add_argument('--deletion-predictions', type=str, default=None,
                       help='Path to deletion error prediction JSONL file (frame-wise predictions)')
    parser.add_argument('--frame-duration', type=float, default=0.08,
                       help='Duration of each frame in seconds for deletion detection (default: 0.08)')
    
    args = parser.parse_args()
    
    print("Loading data...")
    transcription = load_transcription(args.transcription)
    recog = load_recog(args.recog)
    timestamps = load_token_timestamps(args.token_timestamps)
    
    # Load classifier predictions from two sources
    hearing_fault_preds = {}
    intelligibility_preds = {}
    all_classifier_preds = {}
    
    if args.hearing_fault_predictions:
        print(f"Loading hearing fault predictions from {args.hearing_fault_predictions}...")
        hearing_fault_preds = load_classifier_predictions(args.hearing_fault_predictions)
        for file_id, pred_data in hearing_fault_preds.items():
            all_classifier_preds[file_id] = pred_data
    
    if args.intelligibility_predictions:
        print(f"Loading intelligibility predictions from {args.intelligibility_predictions}...")
        intelligibility_preds = load_classifier_predictions(args.intelligibility_predictions)
        # Merge with hearing fault predictions (if any file predicts error, mark as error)
        for file_id, pred_data in intelligibility_preds.items():
            if file_id not in all_classifier_preds:
                all_classifier_preds[file_id] = pred_data
            else:
                # Combine predictions: use max (if any file predicts error, mark as error)
                existing_preds = all_classifier_preds[file_id]['predictions']
                new_preds = pred_data['predictions']
                combined = [max(existing_preds[i], new_preds[i]) 
                           for i in range(min(len(existing_preds), len(new_preds)))]
                all_classifier_preds[file_id]['predictions'] = combined
    
    if not args.hearing_fault_predictions and not args.intelligibility_predictions:
        print("Error: At least one of --hearing-fault-predictions or --intelligibility-predictions must be provided")
        return
    
    # Load deletion predictions if provided
    deletion_preds = {}
    if args.deletion_predictions:
        print(f"Loading deletion predictions from {args.deletion_predictions}...")
        deletion_preds = load_classifier_predictions(args.deletion_predictions)
        print(f"Loaded deletion predictions for {len(deletion_preds)} files")
    
    print(f"Loaded {len(transcription)} transcriptions, {len(recog)} recognitions, "
          f"{len(timestamps)} timestamp files, {len(all_classifier_preds)} prediction files")
    
    # Process each file
    all_metrics = []
    output_lines = []
    file_id_to_output = {}  # Map file_id to output line for sorting
    
    for file_id in transcription.keys():
        if file_id not in recog:
            print(f"Warning: {file_id} not found in recog file")
            continue
        if file_id not in timestamps:
            print(f"Warning: {file_id} not found in timestamps file")
            continue
        if file_id not in all_classifier_preds:
            print(f"Warning: {file_id} not found in classifier predictions")
            continue
        
        # Remove all punctuation from text strings for alignment/error detection
        ref_text = re.sub(r'[^\w\s]', '', transcription[file_id])
        hyp_text_cleaned = re.sub(r'[^\w\s]', '', recog[file_id])
        tokens = timestamps[file_id]
        classifier_preds = all_classifier_preds[file_id]['predictions']
        
        # Tokenize cleaned text (no punctuation) for alignment
        ref_tokens = tokenize_text(ref_text)
        hyp_tokens_cleaned = tokenize_text(hyp_text_cleaned)
        
        # Tokenize original text (with punctuation) for output
        original_hyp_tokens = tokenize_text(recog[file_id])
        
        # Create mapping from cleaned token indices to original token indices
        cleaned_to_original_map = {}
        orig_idx = 0
        for cleaned_idx, cleaned_token in enumerate(hyp_tokens_cleaned):
            # Find matching token in original list (skip punctuation tokens)
            while orig_idx < len(original_hyp_tokens):
                orig_token = original_hyp_tokens[orig_idx]
                if is_punctuation_only(orig_token):
                    orig_idx += 1
                    continue
                # Match tokens (case-insensitive, punctuation removed)
                orig_token_cleaned = re.sub(r'[^\w\s]', '', orig_token).lower()
                if cleaned_token.lower() == orig_token_cleaned:
                    cleaned_to_original_map[cleaned_idx] = orig_idx
                    orig_idx += 1
                    break
                orig_idx += 1
        
        # Align to find ground truth errors (using cleaned tokens)
        substitutions, deletions, insertions, alignment = align_texts(ref_tokens, hyp_tokens_cleaned)
        
        # Detect deletions using deletion predictions (if provided)
        predicted_deletion_frames = []
        if args.deletion_predictions and file_id in deletion_preds:
            deletion_preds_for_file = deletion_preds[file_id]['predictions']
            predicted_deletion_frames = detect_deletions(
                deletion_preds_for_file, tokens, args.frame_duration, tolerance_frames=1
            )
        
        # Get word timestamps from original recognized text (with punctuation for accurate mapping)
        word_timestamps = tokens_to_words_improved(tokens, recog[file_id])
        
        # Filter out punctuation-only words from timestamps
        word_timestamps = [(word, start, end) for word, start, end in word_timestamps 
                          if not is_punctuation_only(word)]
        
        if len(word_timestamps) != len(hyp_tokens_cleaned):
            print(f"Warning: Word count mismatch for {file_id}: "
                  f"{len(word_timestamps)} words from timestamps vs {len(hyp_tokens_cleaned)} from text")
            # Use text-based word count
            word_timestamps = [(word, 0.0, 0.0) for word in hyp_tokens_cleaned]
        # Map token-level predictions to words from each source separately
        # This allows us to determine which error type (unclear/unknown) each word has
        hearing_fault_word_preds = {}
        intelligibility_word_preds = {}
        
        if file_id in hearing_fault_preds:
            hearing_fault_preds_for_file = hearing_fault_preds[file_id]['predictions']
            hearing_fault_word_preds = map_tokens_to_words(hearing_fault_preds_for_file, tokens, word_timestamps)
        
        if file_id in intelligibility_preds:
            intelligibility_preds_for_file = intelligibility_preds[file_id]['predictions']
            intelligibility_word_preds = map_tokens_to_words(intelligibility_preds_for_file, tokens, word_timestamps)
        
        # Combine predictions (if any source predicts error, mark as error)
        word_predictions = {}
        for word_idx in set(list(hearing_fault_word_preds.keys()) + list(intelligibility_word_preds.keys())):
            hearing_pred = hearing_fault_word_preds.get(word_idx, 0)
            intelligibility_pred = intelligibility_word_preds.get(word_idx, 0)
            word_predictions[word_idx] = max(hearing_pred, intelligibility_pred)
        
        # Identify ground truth error word indices
        gt_error_indices = set()
        for sub in substitutions:
            if sub[2] is not None:
                gt_error_indices.add(sub[2])
        for ins in insertions:
            if ins[2] is not None:
                gt_error_indices.add(ins[2])
        
        # Ground truth deletions: these are words in reference that don't appear in hypothesis
        # We need to map them to positions in hypothesis where they should have been
        gt_deletion_positions = []
        # import ipdb
        # ipdb.set_trace()
        for del_item in deletions:
            # del_item[0] is ref_idx, del_item[1] is ref_word
            ref_idx = del_item[0]
            # Find where this deletion should be placed in hypothesis based on alignment
            # A deletion should be placed after the previous matched/substituted word in hypothesis
            hyp_pos = 0  # Default: beginning
            for align_item in alignment:
                # align_item format: (op, ref_idx, hyp_idx, ref_word, hyp_word)
                
                align_ref_idx = align_item[1]
                align_hyp_idx = align_item[2]
                if align_ref_idx is None: # Insertion error, just continue 
                    hyp_pos = align_hyp_idx + 1
                    continue
                if align_ref_idx < ref_idx and align_hyp_idx is not None:
                    # This alignment is before the deletion in reference
                    # Place deletion after this word in hypothesis
                    hyp_pos = align_hyp_idx + 1
                elif align_ref_idx >= ref_idx:
                    # We've passed the deletion position, stop
                    break

            gt_deletion_positions.append(hyp_pos)
        
        # Get predicted error word indices (in cleaned tokens)
        pred_error_indices_cleaned = set()
        for i, pred in word_predictions.items():
            if pred == 1 and i < len(hyp_tokens_cleaned):
                pred_error_indices_cleaned.add(i)
        
        # Map cleaned error indices to original token indices and determine error types
        pred_error_indices = set()
        error_type_map = {}  # Maps original word index to error type ('unclear' or 'unknown')
        
        for cleaned_idx in pred_error_indices_cleaned:
            if cleaned_idx in cleaned_to_original_map:
                orig_idx = cleaned_to_original_map[cleaned_idx]
                pred_error_indices.add(orig_idx)
                
                # Determine error type: prioritize intelligibility (unknown) if both predict error
                hearing_pred = hearing_fault_word_preds.get(cleaned_idx, 0)
                intelligibility_pred = intelligibility_word_preds.get(cleaned_idx, 0)
                
                if intelligibility_pred == 1:
                    error_type_map[orig_idx] = 'unknown'
                elif hearing_pred == 1:
                    error_type_map[orig_idx] = 'unclear'
                else:
                    # Default to 'unknown' if somehow neither predicts error (shouldn't happen)
                    error_type_map[orig_idx] = 'unknown'
        
        # Map ground truth error indices from cleaned to original
        gt_error_indices_original = set()
        for cleaned_idx in gt_error_indices:
            if cleaned_idx in cleaned_to_original_map:
                gt_error_indices_original.add(cleaned_to_original_map[cleaned_idx])
        
        # Map predicted deletion frames to word positions (in cleaned tokens)
        # Convert frame indices to time, then find closest word position
        pred_deletion_positions_cleaned = []
        for frame_idx in predicted_deletion_frames:
            frame_time = frame_idx * args.frame_duration
            # Find the word position where this deletion should be inserted
            # Insert before the word that starts after this time
            insertion_pos = len(hyp_tokens_cleaned)  # Default: end of text
            for word_idx, (word, start_time, end_time) in enumerate(word_timestamps):
                if start_time > frame_time:
                    insertion_pos = word_idx
                    break
            pred_deletion_positions_cleaned.append(insertion_pos)
        
        # Map deletion positions from cleaned to original tokens
        pred_deletion_positions = []
        for cleaned_pos in pred_deletion_positions_cleaned:
            # Find the corresponding original position
            # If deletion is at position i in cleaned, find where it should be in original
            if cleaned_pos < len(hyp_tokens_cleaned):
                # Find the original position after the word at cleaned_pos
                if cleaned_pos in cleaned_to_original_map:
                    orig_pos = cleaned_to_original_map[cleaned_pos]
                else:
                    # Find the closest original position
                    orig_pos = cleaned_pos
                    for i in range(cleaned_pos, -1, -1):
                        if i in cleaned_to_original_map:
                            orig_pos = cleaned_to_original_map[i] + 1
                            break
                pred_deletion_positions.append(orig_pos)
            else:
                # Deletion at the end
                pred_deletion_positions.append(len(original_hyp_tokens))
        
        # Map ground truth deletion positions from cleaned to original
        gt_deletion_positions_original = []
        for cleaned_pos in gt_deletion_positions:
            if cleaned_pos < len(hyp_tokens_cleaned):
                if cleaned_pos in cleaned_to_original_map:
                    orig_pos = cleaned_to_original_map[cleaned_pos]
                else:
                    orig_pos = cleaned_pos
                    for i in range(cleaned_pos, -1, -1):
                        if i in cleaned_to_original_map:
                            orig_pos = cleaned_to_original_map[i] + 1
                            break
                gt_deletion_positions_original.append(orig_pos)
            else:
                gt_deletion_positions_original.append(len(original_hyp_tokens))
        
        # Count words for metrics (use cleaned tokens)
        word_count = len(hyp_tokens_cleaned)
        
        # Compute metrics for substitution/insertion errors (using cleaned indices)
        metrics = compute_metrics(gt_error_indices, pred_error_indices_cleaned, word_count)
        
        # Compute deletion metrics (using original positions for comparison)
        gt_deletion_set = set(gt_deletion_positions_original)
        pred_deletion_set = set(pred_deletion_positions)
        deletion_tp = len(gt_deletion_set & pred_deletion_set)
        deletion_fp = len(pred_deletion_set - gt_deletion_set)
        deletion_fn = len(gt_deletion_set - pred_deletion_set)
        deletion_precision = deletion_tp / (deletion_tp + deletion_fp) if (deletion_tp + deletion_fp) > 0 else 0.0
        deletion_recall = deletion_tp / (deletion_tp + deletion_fn) if (deletion_tp + deletion_fn) > 0 else 0.0
        
        # Compute overall error metrics (including deletions if available)
        if args.deletion_predictions:
            # For overall metrics, we need to combine word errors and deletions
            # Since deletions are at word positions, we can combine them directly
            # Use a special marker for deletions in the error sets (e.g., negative indices or offset)
            # For simplicity, we'll use word_count + deletion_position as the index for deletions
            # Use cleaned indices for word errors
            overall_gt_errors = set(gt_error_indices)  # Already cleaned indices
            overall_pred_errors = set(pred_error_indices_cleaned)  # Use cleaned indices
            
            # Add deletions to overall error sets (offset by word_count to avoid conflicts)
            # Use cleaned positions for overall metrics
            for del_pos in gt_deletion_positions:  # Already cleaned positions
                overall_gt_errors.add(word_count + del_pos)  # Offset to distinguish from word errors
            for del_pos in pred_deletion_positions_cleaned:
                overall_pred_errors.add(word_count + del_pos)
            
            # Compute overall metrics (including deletions)
            overall_tp = len(overall_gt_errors & overall_pred_errors)
            overall_fp = len(overall_pred_errors - overall_gt_errors)
            overall_fn = len(overall_gt_errors - overall_pred_errors)
            overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
            overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
            overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        else:
            # If no deletion predictions, overall metrics are the same as word error metrics
            overall_tp = metrics['true_positives']
            overall_fp = metrics['false_positives']
            overall_fn = metrics['false_negatives']
            overall_precision = metrics['precision']
            overall_recall = metrics['error_recall']
            overall_f1 = metrics['f1']
        
        metrics['file_id'] = file_id
        metrics['deletion_tp'] = deletion_tp
        metrics['deletion_fp'] = deletion_fp
        metrics['deletion_fn'] = deletion_fn
        metrics['deletion_precision'] = deletion_precision
        metrics['deletion_recall'] = deletion_recall
        metrics['overall_tp'] = overall_tp
        metrics['overall_fp'] = overall_fp
        metrics['overall_fn'] = overall_fn
        metrics['overall_precision'] = overall_precision
        metrics['overall_recall'] = overall_recall
        metrics['overall_f1'] = overall_f1
        all_metrics.append(metrics)
        
        # Mark errors and deletions in original text (lowercased but with punctuation)
        marked_text = mark_errors_in_text(original_hyp_tokens, pred_error_indices, pred_deletion_positions, error_type_map)
        output_line = f"{file_id}\t{marked_text}"
        output_lines.append(output_line)
        file_id_to_output[file_id] = output_line
    
    # Write output
    print(f"\nWriting marked text to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    # Sort by F1 score and output top 200 sentences
    if all_metrics:
        # Sort metrics by f1 (descending)
        sorted_metrics = sorted(all_metrics, key=lambda x: x.get('f1', 0.0), reverse=True)
        
        # Get top 200
        top_200 = sorted_metrics[:200]
        
        # Create output file for top 200
        top_200_output_file = args.output + '.top200_f1'
        print(f"\nWriting top 200 sentences by F1 score to {top_200_output_file}...")
        with open(top_200_output_file, 'w', encoding='utf-8') as f:
            for metrics in top_200:
                file_id = metrics['file_id']
                f1 = metrics.get('f1', 0.0)
                precision = metrics.get('precision', 0.0)
                error_recall = metrics.get('error_recall', 0.0)
                if file_id in file_id_to_output:
                    # Write with F1, precision, and recall as comment
                    f.write(f"{file_id_to_output[file_id]}\t# f1={f1:.4f} precision={precision:.4f} recall={error_recall:.4f}\n")
        
        print(f"Top 200 sentences written. F1 score range: {top_200[-1].get('f1', 0.0):.4f} - {top_200[0].get('f1', 0.0):.4f}")
    
        # Compute aggregate metrics
        if all_metrics:
            total_tp = sum(m['true_positives'] for m in all_metrics)
            total_fp = sum(m['false_positives'] for m in all_metrics)
            total_fn = sum(m['false_negatives'] for m in all_metrics)
            total_tn = sum(m['true_negatives'] for m in all_metrics)
            
            agg_false_alarm = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
            agg_error_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            agg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            agg_f1 = 2 * agg_precision * agg_error_recall / (agg_precision + agg_error_recall) if (agg_precision + agg_error_recall) > 0 else 0.0
            
            # Aggregate deletion metrics
            total_del_tp = sum(m.get('deletion_tp', 0) for m in all_metrics)
            total_del_fp = sum(m.get('deletion_fp', 0) for m in all_metrics)
            total_del_fn = sum(m.get('deletion_fn', 0) for m in all_metrics)
            agg_del_precision = total_del_tp / (total_del_tp + total_del_fp) if (total_del_tp + total_del_fp) > 0 else 0.0
            agg_del_recall = total_del_tp / (total_del_tp + total_del_fn) if (total_del_tp + total_del_fn) > 0 else 0.0
            
            # Aggregate overall metrics (including deletions)
            total_overall_tp = sum(m.get('overall_tp', m.get('true_positives', 0)) for m in all_metrics)
            total_overall_fp = sum(m.get('overall_fp', m.get('false_positives', 0)) for m in all_metrics)
            total_overall_fn = sum(m.get('overall_fn', m.get('false_negatives', 0)) for m in all_metrics)
            agg_overall_precision = total_overall_tp / (total_overall_tp + total_overall_fp) if (total_overall_tp + total_overall_fp) > 0 else 0.0
            agg_overall_recall = total_overall_tp / (total_overall_tp + total_overall_fn) if (total_overall_tp + total_overall_fn) > 0 else 0.0
            agg_overall_f1 = 2 * agg_overall_precision * agg_overall_recall / (agg_overall_precision + agg_overall_recall) if (agg_overall_precision + agg_overall_recall) > 0 else 0.0
            
            print(f"\n{'='*60}")
            print("Aggregate Metrics")
            print(f"{'='*60}")
            print(f"False-Alarm Rate: {agg_false_alarm:.4f}")
            print(f"Error Recall (word errors only): {agg_error_recall:.4f}")
            print(f"Precision (word errors only):    {agg_precision:.4f}")
            print(f"F1 Score (word errors only):     {agg_f1:.4f}")
            print(f"\nTotal TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
            
            print(f"\nOverall Metrics (including deletions):")
            print(f"Overall Precision: {agg_overall_precision:.4f}")
            print(f"Overall Recall:    {agg_overall_recall:.4f}")
            print(f"Overall F1:        {agg_overall_f1:.4f}")
            print(f"Overall TP: {total_overall_tp}, FP: {total_overall_fp}, FN: {total_overall_fn}")
            
            if args.deletion_predictions:
                print(f"\nDeletion Metrics:")
                print(f"Deletion Precision: {agg_del_precision:.4f}")
                print(f"Deletion Recall:    {agg_del_recall:.4f}")
                print(f"Deletion TP: {total_del_tp}, FP: {total_del_fp}, FN: {total_del_fn}")
        
        # Write metrics to file if requested
        if args.output_metrics:
            metrics_dict = {
                'aggregate': {
                    'false_alarm_rate': agg_false_alarm,
                    'error_recall': agg_error_recall,
                    'precision': agg_precision,
                    'f1': agg_f1,
                    'true_positives': total_tp,
                    'false_positives': total_fp,
                    'false_negatives': total_fn,
                    'true_negatives': total_tn
                },
                'deletion_aggregate': {
                    'deletion_precision': agg_del_precision,
                    'deletion_recall': agg_del_recall,
                    'deletion_tp': total_del_tp,
                    'deletion_fp': total_del_fp,
                    'deletion_fn': total_del_fn
                } if args.deletion_predictions else {},
                'overall_aggregate': {
                    'overall_precision': agg_overall_precision,
                    'overall_recall': agg_overall_recall,
                    'overall_f1': agg_overall_f1,
                    'overall_tp': total_overall_tp,
                    'overall_fp': total_overall_fp,
                    'overall_fn': total_overall_fn
                },
                'per_file': all_metrics
            }
            with open(args.output_metrics, 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, indent=2)
            print(f"\nMetrics written to {args.output_metrics}")
    
    print(f"\nProcessed {len(all_metrics)} files")


if __name__ == '__main__':
    main()
