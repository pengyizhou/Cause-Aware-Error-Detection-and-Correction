#!/usr/bin/env python3
"""
Prepare batch JSONL file for GPT batch API (user response simulation).

Given reference text and clarification questions, prepare batch requests
for the LLM to act as the user and provide clarifications based on what was actually said.

Output: Batch JSONL file with format:
  {"custom_id": "uttid", "method": "POST", "url": "/v1/responses", "body": {...}}
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional


SYSTEM_PROMPT = """You are simulating a user who is responding to clarification questions about their speech.

Your task is to answer clarification questions as the user would, based on what they actually said (the reference text).
If you are answering two questions, please make them fluent and natural.

Guidelines:
- Answer naturally and concisely, as a real user would
- Use the reference text to determine what the user actually said
- If asked to confirm or choose between options, pick the correct one based on the reference, but DO NOT response the word only. You should form a complete response. 
- If the pronunciation is similar between the options, please spell characters one by one to confirm, or if simple explain the word is easier to say, then explain the word.
- If asked to spell something, spell it letter by letter
- If asked to repeat something, repeat the exact phrase from the reference
- Keep responses brief and conversational, but don't miss important response content, e.g., only reply one of the questions, or only spell one of the corresponding words.
- Do not include explanations or meta-commentary, just answer as the user would

Final output format requirements:
- Please only uppercase the character-by-character spelling of the words, and the first character of the words only when necessary
- Do not uppercase the words you want to quote
""".strip()


USER_PROMPT_TEMPLATE = """You are the user who said the following (this is what you actually said - the reference):

REFERENCE TEXT:
{reference}

The ASR system heard this (with some uncertainty):
ORIGINAL ASR:
{original_asr}

The system is asking you these clarification questions:
CLARIFICATION QUESTION:
{clarification_question}
""".strip()


def parse_text_file(file_path: Path) -> Dict[str, str]:
    """
    Parse a text file with format: UTTID TEXT
    
    Returns:
        Dictionary mapping UTTID to text content
    """
    result = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Split on first space to separate UTTID from text
            parts = line.split(None, 1)
            if len(parts) < 2:
                print(f"[WARN] {file_path.name} Line {line_num}: missing text after UTTID", file=sys.stderr)
                continue
            uttid = parts[0]
            text = parts[1]
            if uttid in result:
                print(f"[WARN] {file_path.name} Line {line_num}: duplicate UTTID {uttid}", file=sys.stderr)
            result[uttid] = text
    return result


def process_files(
    reference_path: Path,
    clarification_path: Path,
    original_asr_path: Path,
    output_path: Path,
    model: str = "gpt-5.2-2025-12-11",
) -> None:
    """
    Process three input text files and generate batch JSONL file for GPT batch API.
    
    Args:
        reference_path: Path to reference text file (UTTID REFERENCE_TEXT)
        clarification_path: Path to clarification questions file (UTTID QUESTION)
        original_asr_path: Path to original ASR file (UTTID ASR_TEXT)
        output_path: Path to output batch JSONL file
        model: Model name (default: gpt-5.2-2025-12-11)
    """
    print(f"[INFO] Reading reference from {reference_path}", file=sys.stderr)
    print(f"[INFO] Reading clarification questions from {clarification_path}", file=sys.stderr)
    print(f"[INFO] Reading original ASR from {original_asr_path}", file=sys.stderr)
    
    # Parse all three files
    references = parse_text_file(reference_path)
    clarifications = parse_text_file(clarification_path)
    original_asrs = parse_text_file(original_asr_path)
    
    # Get all unique UTTIDs
    all_uttids = set(references.keys()) | set(clarifications.keys()) | set(original_asrs.keys())
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    skipped = 0
    
    with output_path.open("w", encoding="utf-8") as out_f:
        for uttid in sorted(all_uttids):
            reference = references.get(uttid, "")
            clarification_question = clarifications.get(uttid, "")
            original_asr = original_asrs.get(uttid, "")
            
            if not reference:
                print(f"[WARN] UTTID {uttid}: missing reference", file=sys.stderr)
                skipped += 1
                continue
            
            if not clarification_question:
                print(f"[WARN] UTTID {uttid}: missing clarification_question", file=sys.stderr)
                skipped += 1
                continue
            
            # Create user_content dict to be JSON stringified
            user_content = {
                "reference": reference,
                "original_asr": original_asr,
                "clarification_question": clarification_question,
            }
            
            # Create batch request format for OpenAI batch API
            batch_request = {
                "custom_id": uttid,  # Use uttid as custom_id for tracking
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": json.dumps(user_content, ensure_ascii=False)},
                    ],
                    "reasoning": {"effort": "medium"}
                }
            }
            
            # Write as JSONL
            out_f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
            count += 1
    
    print(f"[OK] Generated {count} batch requests", file=sys.stderr)
    if skipped > 0:
        print(f"[WARN] Skipped {skipped} entries due to missing data", file=sys.stderr)
    print(f"[OK] Batch JSONL file saved to: {output_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Prepare batch JSONL file for GPT batch API (user response simulation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python3 prepare_batch_user_response.py \\
    --reference gigaspeech_reference.txt \\
    --clarification round_1_to_round2_clarification_questions.txt \\
    --original-asr round_1_final_asr_fixed.txt \\
    --output batch_requests.jsonl \\
    --model "gpt-5.2-2025-12-11"
"""
    )
    ap.add_argument("--reference", type=Path, required=True,
                    help="Reference text file (format: UTTID REFERENCE_TEXT)")
    ap.add_argument("--clarification", type=Path, required=True,
                    help="Clarification questions file (format: UTTID QUESTION)")
    ap.add_argument("--original-asr", type=Path, required=True,
                    help="Original ASR file (format: UTTID ASR_TEXT)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output batch JSONL file for batch API")
    ap.add_argument("--model", type=str, default="gpt-5.2-2025-12-11",
                    help="Model name (default: gpt-5.2-2025-12-11)")
    
    args = ap.parse_args()
    
    print(f"[INFO] Model: {args.model}", file=sys.stderr)
    
    process_files(
        reference_path=args.reference,
        clarification_path=args.clarification,
        original_asr_path=args.original_asr,
        output_path=args.output,
        model=args.model,
    )


if __name__ == "__main__":
    main()

