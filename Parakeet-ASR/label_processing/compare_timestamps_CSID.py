#!/usr/bin/env python3
"""
Script to compare clean (reference) and final (hypothesis) token timestamps,
identify Substitution/Deletion/Insertion errors, and output error labels.
For Deletion error, we distinguish it from other errors by label 2.

Output format:
    uttid\t[start_offset:label,start_offset:label,...]
    where label is 0 for Correct, 1 for Substitution/Insertion/Deletion error

    With --full_range option:
    uttid\t[frame:label,frame:label,...]
    This expands each token to all frames from start_offset to end_offset (inclusive)
"""

import argparse
import ast
import sys
import re
from typing import List, Tuple, Dict, Any


def parse_token_line(line: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Parse a line from the timestamp file.
    Returns (uttid, list_of_tokens).
    """
    parts = line.strip().split('\t', 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid line format: {line[:100]}")
    
    uttid = parts[0]
    tokens_str = parts[1]
    
    # Parse the Python list literal
    try:
        tokens = ast.literal_eval(tokens_str)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Failed to parse tokens: {e}")
    
    return uttid, tokens


def get_token_text(token: Dict[str, Any]) -> str:
    """Extract the token text from a token dict."""
    char_list = token.get('char', [])
    if char_list == []:
        char_list = [token.get('word', '')]
    if isinstance(char_list, list) and len(char_list) > 0:
        return ''.join(str(c) for c in char_list)
    return ''


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing and removing all punctuation."""
    # Lowercase
    text = text.lower()
    # Remove all punctuation (keep only alphanumeric and whitespace)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def align_sequences(ref_tokens: List[Dict], hyp_tokens: List[Dict]) -> List[Tuple[str, int, int]]:
    """
    Align reference and hypothesis tokens using edit distance (Levenshtein distance)
    via dynamic programming. This is the same algorithm used for Word Error Rate (WER) calculation.
    
    The algorithm finds the minimum edit distance alignment between reference and hypothesis,
    considering three operations:
    - Match: tokens are identical (cost = 0)
    - Substitution: tokens differ (cost = 1)
    - Deletion: token in ref but not in hyp (cost = 1)
    - Insertion: token in hyp but not in ref (cost = 1)
    
    Returns a list of (operation, ref_idx, hyp_idx) tuples:
    - ('match', ref_idx, hyp_idx): tokens match
    - ('sub', ref_idx, hyp_idx): substitution error
    - ('del', ref_idx, -1): deletion error
    - ('ins', -1, hyp_idx): insertion error
    """
    ref_texts = [get_token_text(t) for t in ref_tokens]
    hyp_texts = [get_token_text(t) for t in hyp_tokens]
    
    # Normalize texts: lowercase and remove punctuation for comparison
    ref_texts_normalized = [normalize_text(text) for text in ref_texts]
    hyp_texts_normalized = [normalize_text(text) for text in hyp_texts]
    
    # Build alignment using dynamic programming (edit distance / WER algorithm)
    # dp[i][j] = minimum edit distance between ref[0:i] and hyp[0:j]
    m, n = len(ref_texts_normalized), len(hyp_texts_normalized)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    path = [[None] * (n + 1) for _ in range(m + 1)]
    
    # Initialize base cases: converting empty string to string of length i/j
    for i in range(m + 1):
        dp[i][0] = i  # i deletions needed
        if i > 0:
            path[i][0] = 'del'
    for j in range(n + 1):
        dp[0][j] = j  # j insertions needed
        if j > 0:
            path[0][j] = 'ins'
    
    # Fill DP table: standard edit distance recurrence
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Option 1: Match or substitution (diagonal move)
            # Compare normalized texts
            if ref_texts_normalized[i-1] == hyp_texts_normalized[j-1]:
                cost = dp[i-1][j-1]  # Match: no cost
                op = 'match'
            else:
                cost = dp[i-1][j-1] + 1  # Substitution: cost = 1
                op = 'sub'
            
            # Option 2: Deletion (move up: delete ref token)
            if dp[i-1][j] + 1 < cost:
                cost = dp[i-1][j] + 1
                op = 'del'
            
            # Option 3: Insertion (move left: insert hyp token)
            if dp[i][j-1] + 1 < cost:
                cost = dp[i][j-1] + 1
                op = 'ins'
            
            dp[i][j] = cost
            path[i][j] = op
    
    # Backtrack to reconstruct the optimal alignment path
    # This gives us the sequence of operations (match/sub/del/ins) that minimizes edit distance
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i == 0:
            # Only insertions left
            alignment.append(('ins', -1, j - 1))
            j -= 1
        elif j == 0:
            # Only deletions left
            alignment.append(('del', i - 1, -1))
            i -= 1
        else:
            op = path[i][j]
            if op == 'match' or op == 'sub':
                alignment.append((op, i - 1, j - 1))
                i -= 1
                j -= 1
            elif op == 'del':
                alignment.append((op, i - 1, -1))
                i -= 1
            else:  # ins
                alignment.append((op, -1, j - 1))
                j -= 1
    
    alignment.reverse()
    return alignment


def process_files(clean_file: str, final_file: str, output_file: str, full_range: bool = False):
    """Process both files and generate output.
    
    Args:
        clean_file: Path to clean (reference) token timestamps
        final_file: Path to final (hypothesis) token timestamps
        output_file: Path to output file
        full_range: If True, expand labels to all frames from start_offset to end_offset
    """
    # Read both files
    clean_data = {}
    final_data = {}
    
    print(f"Reading {clean_file}...", file=sys.stderr)
    with open(clean_file, 'r', encoding='utf-8') as f:
        for line in f:
            uttid, tokens = parse_token_line(line)
            clean_data[uttid] = tokens
    
    print(f"Reading {final_file}...", file=sys.stderr)
    with open(final_file, 'r', encoding='utf-8') as f:
        for line in f:
            uttid, tokens = parse_token_line(line)
            final_data[uttid] = tokens
    
    # Process each utterance
    print(f"Processing utterances...", file=sys.stderr)
    with open(output_file, 'w', encoding='utf-8') as out:
        # Process all utterances that appear in both files
        all_uttids = set(clean_data.keys()) | set(final_data.keys())
        
        for uttid in sorted(all_uttids):
            if uttid not in clean_data:
                print(f"Warning: {uttid} not in clean file, skipping", file=sys.stderr)
                continue
            if uttid not in final_data:
                print(f"Warning: {uttid} not in final file, skipping", file=sys.stderr)
                continue
            
            ref_tokens = clean_data[uttid]
            hyp_tokens = final_data[uttid]
            
            # Align sequences
            alignment = align_sequences(ref_tokens, hyp_tokens)
            
            # Collect all token ranges with their labels
            token_ranges = []  # [(start_offset, end_offset, label), ...]
            # import ipdb
            # ipdb.set_trace()
            for op, ref_idx, hyp_idx in alignment:
                if op == 'match':
                    # Correct token - use final (hypothesis) start_offset
                    token = hyp_tokens[hyp_idx]
                    start_offset = token.get('start_offset', 0)
                    end_offset = token.get('end_offset', start_offset)
                    label = 0  # 0 = correct
                elif op == 'sub':
                    # Substitution error - use final (hypothesis) start_offset
                    token = hyp_tokens[hyp_idx]
                    start_offset = token.get('start_offset', 0)
                    end_offset = token.get('end_offset', start_offset)
                    label = 1  # 1 = substitution error
                elif op == 'ins':
                    # Insertion error - use final (hypothesis) start_offset
                    token = hyp_tokens[hyp_idx]
                    start_offset = token.get('start_offset', 0)
                    end_offset = token.get('end_offset', start_offset)
                    label = 1  # 1 = insertion error
                elif op == 'del':
                    # Deletion error - use clean (reference) start_offset
                    token = ref_tokens[ref_idx]
                    start_offset = token.get('start_offset', 0)
                    end_offset = token.get('end_offset', start_offset)
                    label = 2  # descrimination between deletion and other errors
                
                token_ranges.append((start_offset, end_offset, label))
            
            # Sort by start_offset (temporal order)
            token_ranges.sort(key=lambda x: x[0])
            
            # Expand to frame-level events
            events = []
            num_tokens = len(token_ranges)
            for idx, (start_offset, end_offset, label) in enumerate(token_ranges):
                if full_range:
                    # For all tokens except the last, use exclusive end to avoid overlap
                    # with the next token's start_offset (keeps start_offset labels)
                    is_last_token = (idx == num_tokens - 1)
                    if is_last_token:
                        # Last token: include end_offset
                        for frame in range(start_offset, end_offset + 1):
                            events.append((frame, label))
                    else:
                        # Non-last token: exclude end_offset to avoid duplicates
                        for frame in range(start_offset, end_offset):
                            events.append((frame, label))
                else:
                    events.append((start_offset, label))
            
            # Format output: uttid [start_offset:label ...]
            event_strs = [f"{offset}:{label}" for offset, label in events]
            output_line = f"{uttid}\t[{','.join(event_strs)}]"
            print(output_line, file=out)
    
    print(f"Output written to {output_file}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description='Compare clean and final token timestamps to generate error labels'
    )
    parser.add_argument('clean_file', help='Path to clean (reference) token timestamps')
    parser.add_argument('final_file', help='Path to final (hypothesis) token timestamps')
    parser.add_argument('output_file', help='Path to output file')
    parser.add_argument(
        '--full_range',
        action='store_true',
        help='Expand labels to all frames from start_offset to end_offset (inclusive)'
    )
    
    args = parser.parse_args()
    
    process_files(args.clean_file, args.final_file, args.output_file, args.full_range)


if __name__ == '__main__':
    main()

