#!/usr/bin/env python3
"""
Cleanup script for token_timestamps.txt files.
Filters out lines that have tokens with overlapping offsets (where start_offset 
is earlier than the previous token's end_offset).
Only returns lines where all tokens are correctly ordered without overlap.
"""

import json
import ast
import sys
import argparse


def has_overlapping_tokens(tokens):
    """
    Check if any token has a start_offset earlier than the previous token's start_offset.
    
    Args:
        tokens: List of token dictionaries
        
    Returns:
        True if there are overlapping tokens, False otherwise
    """
    if not tokens or len(tokens) <= 1:
        return False
    
    # Get all start_offsets, shift by one index, and compare entire lists
    start_offsets = [t.get('start_offset', 0) for t in tokens]
    previous_offsets = start_offsets[:-1]  # All except last
    current_offsets = start_offsets[1:]     # All except first
    
    # Check if any current offset is less than previous offset
    return any(curr < prev for curr, prev in zip(current_offsets, previous_offsets)) 

def process_file(input_file, output_file):
    """
    Process the token_timestamps.txt file and write only lines without overlapping tokens.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
    """
    total_lines = 0
    lines_kept = 0
    lines_removed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            
            # Split line into ID and tokens JSON
            parts = line.split('\t', 1)
            if len(parts) != 2:
                # Skip lines without proper format
                lines_removed += 1
                continue
            
            utterance_id = parts[0]
            tokens_json = parts[1]
            
            try:
                # Parse the tokens JSON
                tokens = ast.literal_eval(tokens_json)
                
                # Check if line has overlapping tokens
                if has_overlapping_tokens(tokens):
                    # Skip this line - it has overlapping tokens
                    lines_removed += 1
                else:
                    # Line is valid - write it as-is
                    f_out.write(line + '\n')
                    lines_kept += 1
                
                total_lines += 1
                
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Failed to parse tokens on line {line_num}: {e}", file=sys.stderr)
                # Skip lines with parse errors
                lines_removed += 1
                total_lines += 1
    
    print(f"Processed {total_lines} lines")
    print(f"Kept {lines_kept} lines without overlapping tokens")
    print(f"Removed {lines_removed} lines with overlapping tokens")


def main():
    parser = argparse.ArgumentParser(
        description='Filter token_timestamps.txt to keep only lines without overlapping token offsets'
    )
    parser.add_argument(
        '-i', '--input_file',
        default='Parakeet-ASR/results/libri-train/all_clean/token_timestamps.txt',
        help='Input token_timestamps.txt file'
    )
    parser.add_argument(
        '-o', '--output',
        default='Parakeet-ASR/results/libri-train/all_clean/token_timestamps.txt.nooverlap',
        help='Output file (default: input_file with .nooverlap suffix)',
    )
    parser.add_argument(
        '--in-place',
        action='store_true',
        help='Modify input file in place (creates backup)'
    )
    
    args = parser.parse_args()
    
    # Determine output file
    if args.in_place:
        import shutil
        # Create backup
        backup_file = args.input_file + '.bak'
        shutil.copy2(args.input_file, backup_file)
        print(f"Created backup: {backup_file}")
        output_file = args.input_file
    elif args.output:
        output_file = args.output
    else:
        output_file = args.input_file + '.nooverlap'
    
    print(f"Input:  {args.input_file}")
    print(f"Output: {output_file}")
    print()
    
    process_file(args.input_file, output_file)
    print(f"\nDone! Output written to: {output_file}")


if __name__ == '__main__':
    main()

