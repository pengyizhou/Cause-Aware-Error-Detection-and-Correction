#!/usr/bin/env python3
"""
Extract user response text from batch API responses JSONL file for direct LLM usage (no TTS).

Output format: uttid-roundx\ttext (tab-separated, no quotes, no normalization)
"""

import argparse
import json
import re
import sys
from pathlib import Path

def normalize_text(text: str) -> str:
    """
    Normalize text for TTS:
    - Replace newlines with spaces
    - Remove special characters (quotes, dashes, asterisks, etc.)
    - Replace them with blank space
    - Combine multiple blanks into one blank
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    # Replace newlines and other whitespace characters with space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove specific special characters that might affect TTS
    # Remove: quotes (' " ' "), dashes (- — – —), asterisks (*), 
    # underscores (_), backslashes (\), forward slashes (/), 
    # brackets ([ ] { } ( )), and other special formatting characters
    # Keep: alphanumeric, basic punctuation (.,!?;:), and spaces
    text = re.sub(r"['\"'\"`]", ' ', text)  # Remove quotes
    text = re.sub(r'[-—–—]', ' ', text)  # Remove dashes
    text = re.sub(r'\*+', ' ', text)  # Remove asterisks
    text = re.sub(r'[\[\]{}()]', ' ', text)  # Remove brackets
    text = re.sub(r'[\\/]', ' ', text)  # Remove backslashes and forward slashes
    text = re.sub(r'_+', ' ', text)  # Remove underscores
    text = re.sub(r'…', ' ', text)  # Remove ellipsis
    text = re.sub(r'—', ' ', text)  # Remove em dash
    text = re.sub(r'–', ' ', text)  # Remove en dash
    text = re.sub(r'‑', ' ', text)  # Remove hyphen
    text = re.sub(r'"', ' ', text)  # Remove left double quote
    text = re.sub(r'"', ' ', text)  # Remove right double quote
    text = re.sub(r''', "'", text)  # Remove left single quote
    text = re.sub(r''', "'", text)  # Remove right single quote
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Strip leading/trailing spaces
    text = text.strip()
    
    return text

def extract_text_from_responses(
    input_path: Path,
    output_path: Path,
    round_num: int = 1,
) -> None:
    """
    Extract text from responses JSONL and format as uttid-roundx\ttext (tab-separated).
    
    Args:
        input_path: Path to input JSONL responses file
        output_path: Path to output text file
        round_num: Round number (default: 1)
    """
    print(f"[INFO] Reading responses from {input_path}", file=sys.stderr)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    skipped = 0
    
    with input_path.open("r", encoding="utf-8") as f, \
         output_path.open("w", encoding="utf-8") as out_f:
        
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                obj = json.loads(line)
                custom_id = obj.get("custom_id")
                
                if not custom_id:
                    print(f"[WARN] Line {line_num}: missing custom_id", file=sys.stderr)
                    skipped += 1
                    continue
                
                # Extract text from response.body.output
                # Handle both formats:
                # 1. output[0] is message directly
                # 2. output[0] is reasoning, output[1] is message
                response = obj.get("response", {})
                body = response.get("body", {})
                output = body.get("output", [])
                
                if not output:
                    print(f"[WARN] Line {line_num}: uttid {custom_id} missing output", file=sys.stderr)
                    skipped += 1
                    continue
                
                # Find the message object (skip reasoning objects)
                message_obj = None
                for item in output:
                    if item.get("type") == "message":
                        message_obj = item
                        break
                
                if not message_obj:
                    print(f"[WARN] Line {line_num}: uttid {custom_id} missing message in output", file=sys.stderr)
                    skipped += 1
                    continue
                
                content = message_obj.get("content", [])
                if not content:
                    print(f"[WARN] Line {line_num}: uttid {custom_id} missing content", file=sys.stderr)
                    skipped += 1
                    continue
                
                text = content[0].get("text", "")
                if not text:
                    print(f"[WARN] Line {line_num}: uttid {custom_id} missing text", file=sys.stderr)
                    skipped += 1
                    continue
                
                # Normalize text: replace newlines with spaces and collapse multiple spaces
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = re.sub(r' +', ' ', text)  # Collapse multiple spaces to single space
                text = text.strip()  # Remove leading/trailing whitespace
                # Normalize text for TTS
                normalized_text = normalize_text(text)
                text = normalized_text
                # Use text directly without TTS normalization
                # Format: uttid-roundx\ttext (tab-separated, single line)
                formatted_line = f"{custom_id}-round{round_num}\t{text}\n"
                out_f.write(formatted_line)
                count += 1
                
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {line_num}: failed to parse JSON: {e}", file=sys.stderr)
                skipped += 1
                continue
            except (KeyError, IndexError, TypeError) as e:
                print(f"[WARN] Line {line_num}: failed to extract text: {e}", file=sys.stderr)
                skipped += 1
                continue
    
    print(f"[OK] Extracted {count} responses", file=sys.stderr)
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} entries", file=sys.stderr)
    print(f"[OK] Output saved to: {output_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Extract user response text from batch API responses JSONL (direct LLM usage, no TTS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 extract_user_response_direct.py \\
    --input round_1_user_response.responses.jsonl \\
    --output round_1_user_response.asr.txt \\
    --round 1
"""
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="Input JSONL responses file")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output text file (tab-separated format)")
    ap.add_argument("--round", type=int, default=1,
                    help="Round number (default: 1)")
    
    args = ap.parse_args()
    
    extract_text_from_responses(
        input_path=args.input,
        output_path=args.output,
        round_num=args.round,
    )


if __name__ == "__main__":
    main()

