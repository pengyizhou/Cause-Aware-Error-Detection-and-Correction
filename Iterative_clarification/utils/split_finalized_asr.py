#!/usr/bin/env python3
"""
Split fixed ASR file into finalized and not_finalized files.

Finalized: utterances with <FINAL> tag and no uncertainty tags
Not finalized: utterances without <FINAL> tag or with uncertainty tags
"""

import argparse
import sys
from pathlib import Path


def split_finalized_asr(input_path: Path, finalized_path: Path, not_finalized_path: Path):
    """
    Split ASR file into finalized and not_finalized files.
    
    Args:
        input_path: Input fixed ASR file (format: uttid fixed_asr_text)
        finalized_path: Output file for finalized utterances
        not_finalized_path: Output file for not finalized utterances
    """
    finalized_count = 0
    not_finalized_count = 0
    
    with input_path.open("r", encoding="utf-8") as f, \
         finalized_path.open("w", encoding="utf-8") as finalized_f, \
         not_finalized_path.open("w", encoding="utf-8") as not_finalized_f:
        
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Format: uttid fixed_asr_text
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            
            uttid, asr_text = parts[0], parts[1]
            
            # Check if finalized: has <FINAL> tag and no uncertainty tags
            has_final = "<FINAL>" in asr_text
            has_uncertainty = any(tag in asr_text for tag in ["<unknown>", "<del>", "<unclear>", "</unknown>", "</unclear>"])
            
            if has_final or not has_uncertainty:
                # Finalized: write to finalized file
                finalized_f.write(f"{uttid} {asr_text}\n")
                finalized_count += 1
            else:
                # Not finalized: write to not_finalized file
                not_finalized_f.write(f"{uttid} {asr_text}\n")
                not_finalized_count += 1
    
    print(f"[INFO] Split {finalized_count} finalized and {not_finalized_count} not_finalized utterances", file=sys.stderr)
    print(f"[OK] Finalized file: {finalized_path}", file=sys.stderr)
    print(f"[OK] Not finalized file: {not_finalized_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Split fixed ASR file into finalized and not_finalized files"
    )
    ap.add_argument("--input", type=Path, required=True,
                    help="Input fixed ASR file")
    ap.add_argument("--finalized", type=Path, required=True,
                    help="Output file for finalized utterances")
    ap.add_argument("--not-finalized", type=Path, required=True,
                    help="Output file for not finalized utterances")
    
    args = ap.parse_args()
    
    split_finalized_asr(
        input_path=args.input,
        finalized_path=args.finalized,
        not_finalized_path=args.not_finalized,
    )


if __name__ == "__main__":
    main()

