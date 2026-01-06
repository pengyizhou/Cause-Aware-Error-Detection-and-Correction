#!/usr/bin/env python3
"""
Replace "-" with " " in the content part only, preserving the uttid-roundN prefix.

Input format: uttid-roundN\tcontent or uttid-roundN content
Output format: uttid-roundN\tcontent (with dashes in content replaced by spaces)
"""

import argparse
import re
import sys
from pathlib import Path


def uppercase_single_char_sequences(text: str) -> str:
    """
    Detect sequences of single characters separated by spaces (more than 2 chars)
    and uppercase them if not already all uppercase.
    Don't touch sequences that are followed by "'s" (possessive).
    
    Example: "b a r e I l l y" -> "B A R E I L L Y"
    Example: "i 's" -> stays as "i 's" (not touched)
    """
    # Pattern: single letter, space, single letter, space, ... (at least 3 letters total)
    # Matches sequences like "b a r e I l l y" or "U N A V A G A M"
    # But not if followed by "'s" (possessive)
    pattern = r'\b([a-zA-Z] ){2,}[a-zA-Z](?!\s+\'s\b)'
    
    def replace_match(match):
        sequence = match.group(0)
        # Check if already all uppercase
        if sequence.isupper():
            return sequence
        # Convert to uppercase
        return sequence.upper()
    
    return re.sub(pattern, replace_match, text)


def replace_dash_in_content(input_path: Path, output_path: Path):
    """
    Replace "-" with " " in content only, preserving uttid-roundN.
    Also uppercase single character sequences if not already uppercase.
    
    Args:
        input_path: Input file path
        output_path: Output file path
    """
    count = 0
    
    with input_path.open("r", encoding="utf-8") as f_in, \
         output_path.open("w", encoding="utf-8") as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.rstrip('\n\r')
            if not line:
                f_out.write('\n')
                continue
            
            # Split on tab or first space after uttid-roundN
            # Pattern: uttid-roundN followed by tab or space, then content
            if '\t' in line:
                # Tab-separated format
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    uttid_round, content = parts[0], parts[1]
                    # Replace "-" and "‑" (non-breaking hyphen/en-dash) with " " in content only
                    content = content.replace('-', ' ').replace('‑', ' ')
                    # Remove ellipsis
                    content = content.replace('…', ' ')
                    # Remove special characters like *, and other common special chars
                    content = re.sub(r'[*_\[\]{}()\\/]', ' ', content)
                    # Collapse multiple spaces to single space
                    content = re.sub(r' +', ' ', content).strip()
                    # Uppercase single character sequences if not already uppercase
                    content = uppercase_single_char_sequences(content)
                    f_out.write(f"{uttid_round}\t{content}\n")
                    count += 1
                else:
                    # No content, just write as-is
                    f_out.write(f"{line}\n")
            else:
                # Space-separated format: uttid-roundN content...
                # Match uttid-roundN pattern and split after it
                match = re.match(r'^([^\s]+-round\d+)\s+(.+)$', line)
                if match:
                    uttid_round, content = match.groups()
                    # Replace "-" and "‑" (non-breaking hyphen/en-dash) with " " in content only
                    content = content.replace('-', ' ').replace('‑', ' ')
                    # Remove ellipsis
                    content = content.replace('…', ' ')
                    # Remove special characters like *, and other common special chars
                    content = re.sub(r'[*_\[\]{}()\\/]', ' ', content)
                    # Collapse multiple spaces to single space
                    content = re.sub(r' +', ' ', content).strip()
                    # Uppercase single character sequences if not already uppercase
                    content = uppercase_single_char_sequences(content)
                    f_out.write(f"{uttid_round}\t{content}\n")
                    count += 1
                else:
                    # No match, write as-is
                    f_out.write(f"{line}\n")
    
    print(f"[OK] Processed {count} lines", file=sys.stderr)
    print(f"[OK] Output saved to: {output_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Replace '-' with ' ' in content only, preserving uttid-roundN"
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="Input file path")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output file path")
    
    args = ap.parse_args()
    
    replace_dash_in_content(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

