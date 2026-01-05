#!/usr/bin/env python3
"""
Generate token-level error labels by computing edit distance between ASR output and ground truth.

Input files:
    - timestamps.txt: Word-level ASR output with timestamps
    - token_timestamps.txt: Token/subword-level ASR output with timestamps
    - transcription.txt: Ground truth transcriptions

Output format (matching token_timestamps.txt.nooverlap.labels.txt):
    uttid\t[start_offset:label,start_offset:label,...]
    where label is 0 for Correct (C), 1 for Substitution (S) or Insertion (I)

    With --full_range option:
    uttid\t[frame:label,frame:label,...]
    This expands each token to all frames from start_offset to end_offset (inclusive)

Usage:
    python generate_token_labels.py \
        --timestamps timestamps.txt \
        --token_timestamps token_timestamps.txt \
        --transcription transcription.txt \
        --output output_labels.txt \
        [--deletion_strategy adjacent|skip|interpolate]
"""

import argparse
import ast
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


def load_timestamps(filepath: str) -> Dict[str, List[Dict]]:
    """Load word-level timestamps from timestamps.txt"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            uttid, timestamps_str = parts
            try:
                timestamps = ast.literal_eval(timestamps_str)
                data[uttid] = timestamps
            except (SyntaxError, ValueError) as e:
                print(f"Warning: Failed to parse timestamps for {uttid}: {e}")
    return data


def load_token_timestamps(filepath: str) -> Dict[str, List[Dict]]:
    """Load token-level timestamps from token_timestamps.txt"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            uttid, timestamps_str = parts
            try:
                timestamps = ast.literal_eval(timestamps_str)
                data[uttid] = timestamps
            except (SyntaxError, ValueError) as e:
                print(f"Warning: Failed to parse token timestamps for {uttid}: {e}")
    return data


def load_transcription(filepath: str) -> Dict[str, str]:
    """Load ground truth transcriptions from transcription.txt"""
    data = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)  # Split on first whitespace
            if len(parts) >= 1:
                uttid = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                data[uttid] = text
    return data


def normalize_word(word: str) -> str:
    """Normalize word for comparison (lowercase, remove punctuation at edges)"""
    word = word.lower().strip()
    # Remove leading/trailing punctuation but keep apostrophes inside words
    word = re.sub(r'^[^\w\']+|[^\w\']+$', '', word)
    return word


def edit_distance_with_alignment(hyp: List[str], ref: List[str]) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """
    Compute edit distance and return alignment operations.
    
    Returns list of tuples: (operation, hyp_index, ref_index)
    - ('C', i, j): Correct - hyp[i] matches ref[j]
    - ('S', i, j): Substitution - hyp[i] should be ref[j]
    - ('I', i, None): Insertion - hyp[i] is extra (not in ref)
    - ('D', None, j): Deletion - ref[j] is missing from hyp
    """
    m, n = len(hyp), len(ref)
    
    # DP table for edit distance
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i  # Deletions from hyp (insertions)
    for j in range(n + 1):
        dp[0][j] = j  # Insertions to hyp (deletions from ref)
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if normalize_word(hyp[i-1]) == normalize_word(ref[j-1]):
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # Insertion (extra word in hyp)
                    dp[i][j-1] + 1,    # Deletion (missing word in hyp)
                    dp[i-1][j-1] + 1   # Substitution
                )
    
    # Backtrack to get alignment
    alignment = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and normalize_word(hyp[i-1]) == normalize_word(ref[j-1]):
            alignment.append(('C', i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append(('S', i-1, j-1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            alignment.append(('I', i-1, None))
            i -= 1
        else:
            alignment.append(('D', None, j-1))
            j -= 1
    
    alignment.reverse()
    return alignment


def get_word_from_timestamps(word_info: Dict) -> str:
    """Extract word from word_info dict"""
    return word_info.get('word', '')


def map_tokens_to_words(token_timestamps: List[Dict], word_timestamps: List[Dict]) -> List[int]:
    """
    Map each token to its corresponding word index based on start_offset overlap.
    Returns a list where token_to_word[i] = word index that token i belongs to.
    """
    if not token_timestamps or not word_timestamps:
        return []
    
    token_to_word = []
    
    for token in token_timestamps:
        token_start = token.get('start_offset', 0)
        token_end = token.get('end_offset', 0)
        
        # Find the word that best overlaps with this token
        best_word_idx = 0
        best_overlap = -1
        
        for word_idx, word_info in enumerate(word_timestamps):
            word_start = word_info.get('start_offset', 0)
            word_end = word_info.get('end_offset', 0)
            
            # Calculate overlap
            overlap_start = max(token_start, word_start)
            overlap_end = min(token_end, word_end)
            overlap = max(0, overlap_end - overlap_start)
            
            # Also check if token start falls within word boundaries
            if word_start <= token_start <= word_end:
                overlap += 10  # Bonus for containment
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_word_idx = word_idx
        
        token_to_word.append(best_word_idx)
    
    return token_to_word


def process_utterance(
    uttid: str,
    word_timestamps: List[Dict],
    token_timestamps: List[Dict],
    ref_text: str,
    deletion_strategy: str = 'adjacent',
    full_range: bool = False
) -> List[Tuple]:
    """
    Process a single utterance and return token labels.
    
    Args:
        uttid: Utterance ID
        word_timestamps: Word-level ASR output
        token_timestamps: Token-level ASR output
        ref_text: Ground truth text
        deletion_strategy: How to handle deletions ('adjacent', 'skip', 'interpolate')
        full_range: If True, expand to all frames from start_offset to end_offset
    
    Returns:
        List of (frame, label) tuples. If full_range=True, expands each token to multiple frames.
    """
    # Extract hypothesis words from ASR output
    hyp_words = [get_word_from_timestamps(w) for w in word_timestamps]
    
    # Extract reference words from ground truth
    ref_words = ref_text.strip().split()
    
    if not hyp_words and not ref_words:
        return []
    
    # Get alignment using edit distance
    alignment = edit_distance_with_alignment(hyp_words, ref_words)
    
    # Create word-level labels (0 for correct, 1 for error)
    word_labels = {}  # hyp_word_idx -> label
    
    # Track deletion positions for adjacent strategy
    deletion_positions = []  # (position_in_hyp, ref_word_idx)
    
    hyp_idx = 0
    for op, h_idx, r_idx in alignment:
        if op == 'C':
            word_labels[h_idx] = 0  # Correct
        elif op == 'S':
            word_labels[h_idx] = 1  # Substitution error
        elif op == 'I':
            word_labels[h_idx] = 1  # Insertion error
        elif op == 'D':
            # Deletion: word missing from hypothesis
            # Record the position where it should have been
            deletion_positions.append((hyp_idx, r_idx))
        
        if op in ['C', 'S', 'I']:
            hyp_idx = h_idx + 1
    
    # Handle deletions based on strategy
    if deletion_strategy == 'adjacent':
        # Mark adjacent words as having nearby deletions (optional: mark them as errors)
        for del_pos, ref_idx in deletion_positions:
            # Mark the word before or after the deletion
            if del_pos > 0 and del_pos - 1 in word_labels:
                # Option: mark preceding word as error due to adjacent deletion
                # word_labels[del_pos - 1] = 1  # Uncomment to mark adjacent as error
                pass
            if del_pos < len(hyp_words) and del_pos in word_labels:
                # Option: mark following word as error due to adjacent deletion
                # word_labels[del_pos] = 1  # Uncomment to mark adjacent as error
                pass
    
    # Map tokens to words
    token_to_word = map_tokens_to_words(token_timestamps, word_timestamps)
    
    # Generate token labels
    token_labels = []
    num_tokens = len(token_timestamps)
    for token_idx, token in enumerate(token_timestamps):
        start_offset = token.get('start_offset', 0)
        end_offset = token.get('end_offset', start_offset)
        
        if token_idx < len(token_to_word):
            word_idx = token_to_word[token_idx]
            label = word_labels.get(word_idx, 0)  # Default to 0 if not found
        else:
            label = 0  # Default
        
        if full_range:
            # For all tokens except the last, use exclusive end to avoid overlap
            # with the next token's start_offset (keeps start_offset labels)
            is_last_token = (token_idx == num_tokens - 1)
            if is_last_token:
                # Last token: include end_offset
                for frame in range(start_offset, end_offset + 1):
                    token_labels.append((frame, label))
            else:
                # Non-last token: exclude end_offset to avoid duplicates
                for frame in range(start_offset, end_offset):
                    token_labels.append((frame, label))
        else:
            token_labels.append((start_offset, label))
    
    return token_labels


def format_output(uttid: str, token_labels: List[Tuple[int, int]]) -> str:
    """Format output in the target format: uttid\t[start:label,start:label,...]"""
    if not token_labels:
        return f"{uttid}\t[]"
    
    labels_str = ",".join(f"{start}:{label}" for start, label in token_labels)
    return f"{uttid}\t[{labels_str}]"


def main():
    parser = argparse.ArgumentParser(
        description='Generate token-level error labels using edit distance alignment'
    )
    parser.add_argument('--timestamps', required=True, help='Path to timestamps.txt')
    parser.add_argument('--token_timestamps', required=True, help='Path to token_timestamps.txt')
    parser.add_argument('--transcription', required=True, help='Path to transcription.txt (ground truth)')
    parser.add_argument('--output', required=True, help='Output file path')
    parser.add_argument(
        '--deletion_strategy',
        choices=['adjacent', 'skip', 'interpolate'],
        default='skip',
        help='''Strategy for handling deletion errors:
            - skip: Don't mark any token for deletions (default)
            - adjacent: Mark tokens adjacent to deletions (not implemented by default)
            - interpolate: Estimate position and mark nearby tokens
        '''
    )
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    parser.add_argument(
        '--full_range',
        action='store_true',
        help='Expand labels to all frames from start_offset to end_offset (inclusive)'
    )
    
    args = parser.parse_args()
    
    # Load data
    if args.verbose:
        print("Loading timestamps...")
    word_timestamps = load_timestamps(args.timestamps)
    
    if args.verbose:
        print("Loading token timestamps...")
    token_timestamps = load_token_timestamps(args.token_timestamps)
    
    if args.verbose:
        print("Loading transcriptions...")
    transcriptions = load_transcription(args.transcription)
    
    if args.verbose:
        print(f"Loaded {len(word_timestamps)} word timestamps")
        print(f"Loaded {len(token_timestamps)} token timestamps")
        print(f"Loaded {len(transcriptions)} transcriptions")
    
    # Process each utterance
    results = []
    processed = 0
    skipped = 0
    
    # Use utterance IDs from token_timestamps as the reference
    for uttid in token_timestamps:
        if uttid not in word_timestamps:
            if args.verbose:
                print(f"Warning: No word timestamps for {uttid}")
            skipped += 1
            continue
        
        if uttid not in transcriptions:
            if args.verbose:
                print(f"Warning: No transcription for {uttid}")
            skipped += 1
            continue
        
        token_labels = process_utterance(
            uttid,
            word_timestamps[uttid],
            token_timestamps[uttid],
            transcriptions[uttid],
            args.deletion_strategy,
            args.full_range
        )
        
        results.append(format_output(uttid, token_labels))
        processed += 1
        
        if args.verbose and processed % 1000 == 0:
            print(f"Processed {processed} utterances...")
    
    # Write output
    with open(args.output, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')
    
    print(f"Done! Processed {processed} utterances, skipped {skipped}")
    print(f"Output written to {args.output}")


if __name__ == '__main__':
    main()
