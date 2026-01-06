#!/usr/bin/env python3
"""
Prepare batch JSONL file for GPT batch API (dialogue/clarification task).

Input format (one per line):
  uttid <ASR_TEXT...>
or JSONL format for Round 2+

Output: Batch JSONL file with format:
  {"custom_id": "uttid", "method": "POST", "url": "/v1/responses", "body": {...}}
"""

from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional


SYSTEM_PROMPT = """You are an ASR-intent clarifier.
The clarification questions you ask should be natural and fluent.
For example, you may ask:
	- I didn't catch this part: xxx. Can you say it again?
	- Just to confirm, did you mean xxx or xxx?
	- Is it xxx? If not, could you rephrase that part?
    - I heard: xxx. Did you mean xxx?
	- You said xxx — are you talking about xxx or xxx?
	- Is xxx actually xxx / xxx / xxx?
	- Is xxx a name? If so, could you spell it letter-by-letter?
	- Can you spell xxx slowly, or give a close synonym?

Please remember to involve some context to remind the user.

The output format should be a json string with the following fields:
- "round": current round number
- "clarification_question": clarification question. You may ask up to two questions but using only one sentence.
- "fixed_asr": fixed ASR (if there is no uncertainty labels in the ASR, directly output the ASR without asking any more questions)
DO NOT CONSIDER ANY POTENTIAL ERRORS IN THE ASR. ONLY FOCUS ON THE UNCERTAINTY LABELS.
Prioritize questions that are essential to preserve/restore the intended meaning.
Keep original words that without any uncertainty labels; only replace uncertain spans with confirmed text.
- Round 1: ask one or two clarification questions
- Round 2-3: ask one or two clarification questions, and fixed ASR using the prior rounds' clarifications. If already clarified all information, directly output <FINAL> fixed ASR without asking any more questions. The output format should be {{"round": current_round, "clarification_question": clarification_question, "fixed_asr": fixed_asr}} or {{"round": current_round, "clarification_question": [], "fixed_asr": fixed_asr}}
- Round 4: output <FINAL> fixed ASR without asking any more questions. The output format should be {{"round": current_round, "clarification_question": [], "fixed_asr": fixed_asr}}

Tag-specific clarification rules:
- User does not know about the tags like <del> <unknown> <unclear>, etc. You should not mention these tags in the clarification question.
- However, in the fixed_asr, you should keep the tags if the word is not clarified by the user unless the word is clarified by the user in CONTEXT.
- <del> or <unclear>: ask the user to repeat that segment; please suggest moving to a quieter place and/or closer to the mic.
- <unknown>: confirm by offering:
   (a) a rephrased guess, and/or
   (b) asking the user to spell the word character-by-character.
- If the unknown/unclear part is just a number, simply confirm with the user.
""".strip()


USER_PROMPT_TEMPLATE = """INPUTS
INITIAL_ASR_TEXT:
{initial_asr_text}

CONTEXT:
{context}

ROUND:
{round_num}

DO NOT REVERSE THE ORDER OF THE WORDS. THE USER RESPONSE IN CONTEXT IS ONLY FOR REFERENCE. 
IT IS ONLY USED TO REFINE THE ASR RESULTS.
""".strip()


def parse_line(line: str):
    """Parse 'uttid ASR_TEXT...' (space or tab separated)."""
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Bad line (need: uttid <ASR_TEXT...>): {line!r}")
    uttid, asr_text = parts[0], parts[1].strip()
    return uttid, asr_text


def load_round1_qa_files(questions_path: Optional[Path], answers_path: Optional[Path]) -> Dict[str, tuple[str, str]]:
    """
    Load Round 1 questions and answers files and create a mapping from uttid to (question, answer).
    
    Args:
        questions_path: Path to questions file (format: uttid question_text)
        answers_path: Path to answers file (format: uttid-round1\tanswer_text)
    
    Returns:
        Dictionary mapping uttid -> (question, answer)
    """
    qa_map = {}
    
    if questions_path is None or answers_path is None:
        return qa_map
    
    # Load questions
    questions_dict = {}
    if questions_path.exists():
        with questions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: uttid question_text
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    uttid, question = parts[0], parts[1]
                    questions_dict[uttid] = question
    
    # Load answers
    answers_dict = {}
    if answers_path.exists():
        with answers_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Format: uttid-roundN\tanswer_text or uttid-roundN answer_text
                # Supports any round number: round1, round2, round3, etc.
                if '\t' in line:
                    parts = line.split('\t', 1)
                else:
                    parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    uttid_round, answer = parts[0], parts[1]
                    # Remove -roundN suffix to get base uttid (handles round1, round2, round3, etc.)
                    match = re.match(r'^(.+)-round(\d+)$', uttid_round)
                    if match:
                        uttid = match.group(1)  # Extract base uttid
                        answers_dict[uttid] = answer
                    else:
                        # Fallback: if no round suffix, use as-is
                        answers_dict[uttid_round] = answer
    
    # Combine questions and answers
    for uttid in questions_dict:
        if uttid in answers_dict:
            qa_map[uttid] = (questions_dict[uttid], answers_dict[uttid])
        else:
            print(f"[WARN] No answer found for uttid {uttid}", file=sys.stderr)
    
    for uttid in answers_dict:
        if uttid not in questions_dict:
            print(f"[WARN] No question found for uttid {uttid}", file=sys.stderr)
    
    print(f"[INFO] Loaded {len(qa_map)} question-answer pairs", file=sys.stderr)
    return qa_map


def process_file(
    input_path: Path,
    output_path: Path,
    context_text: str,
    round_num: int,
    model: str = "gpt-5.2",
    input_format: str = "text",
    qa_map: Optional[Dict[str, tuple[str, str]]] = None,
    original_asr_map: Optional[Dict[str, str]] = None,
) -> None:
    """
    Process input file and generate batch JSONL file for GPT batch API.
    
    Args:
        input_path: Path to input file
        output_path: Path to output batch JSONL file
        context_text: Context text (used for text format, Round 1)
        round_num: Round number
        model: Model name (default: gpt-5.2)
        input_format: "text" or "jsonl"
        qa_map: Dictionary mapping uttid -> (question, answer) for Round 2+ context
        original_asr_map: Dictionary mapping uttid -> original ASR text (optional, for Round 2+)
    """
    print(f"[INFO] Reading input from {input_path} (format: {input_format})", file=sys.stderr)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    
    with input_path.open("r", encoding="utf-8") as f, \
         output_path.open("w", encoding="utf-8") as out_f:
        
        for line_num, line in enumerate(f, 1):
            if input_format == "jsonl":
                # Parse JSONL format (for Round 2+)
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    uttid = record.get("uttid")
                    if not uttid:
                        continue
                    
                    # Extract fields for Round 2
                    original_asr = record.get("original_asr", "")
                    clarification_question_round1 = record.get("clarification_question_round1", "")
                    current_fixed_asr = record.get("current_fixed_asr", "")
                    user_response_asr = record.get("user_response_asr", "")
                    
                    # Format context from Round 2 fields
                    prev_round = round_num - 1
                    context = f"""Round {prev_round}:
- Clarification Question: {clarification_question_round1}
- User Response: {user_response_asr}"""
                    
                    # Use current_fixed_asr as initial ASR text
                    asr_text = current_fixed_asr
                    
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_num}: failed to parse JSON: {e}", file=sys.stderr)
                    continue
            else:
                # Parse text format (for Round 1 or Round 2+)
                parsed = parse_line(line)
                if parsed is None:
                    continue
                uttid, asr_text = parsed
                
                if round_num >= 2 and qa_map is not None and uttid in qa_map:
                    # Use QA mapping for Round 2+ context
                    question, answer = qa_map[uttid]
                    prev_round = round_num - 1
                    
                    context = f"""Round {prev_round}:
- Clarification Question: {question}
- User Response: {answer}"""
                else:
                    # Use provided context_text (for Round 1 or when QA map not available)
                    context = context_text
            
            # Generate user prompt
            # Create user_content dict to be JSON stringified
            user_content = {
                "initial_asr_text": asr_text,
                "context": context,
                "round_num": round_num,
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
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "clarification_result",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "round": {"type": "integer"},
                                    "clarification_question": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "fixed_asr": {"type": "string"}
                                },
                                "required": ["round", "clarification_question", "fixed_asr"]
                            }
                        }
                    },
                    "reasoning": {"effort": "medium"}
                }
            }
            
            # Write as JSONL
            out_f.write(json.dumps(batch_request, ensure_ascii=False) + "\n")
            count += 1
    
    print(f"[OK] Generated {count} batch requests", file=sys.stderr)
    print(f"[OK] Batch JSONL file saved to: {output_path}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description="Prepare batch JSONL file for GPT batch API (dialogue/clarification task)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage (Round 1, text format):
  python prepare_batch_dialogue.py input.txt -o batch_requests.jsonl \\
    --round 1 --model "gpt-5.2"

Example usage (Round 2+, JSONL format):
  python prepare_batch_dialogue.py round2_combined.jsonl -o batch_requests.jsonl \\
    --round 2 --input-format jsonl --model "gpt-5.2"

Example usage (Round 2+, text format with QA files):
  python prepare_batch_dialogue.py round2_input.txt -o batch_requests.jsonl \\
    --round 2 --round1-questions questions.txt --round1-answers answers.txt \\
    --original-asr original_asr.txt --model "gpt-5.2"
"""
    )
    ap.add_argument("input", type=Path, help="Input file: text file (uttid ASR_TEXT...) or JSONL file (for Round 2+)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output batch JSONL path")
    ap.add_argument("--round", dest="round_num", type=int, default=1, choices=[1, 2, 3, 4],
                    help="Clarification round number (default: 1)")
    ap.add_argument("--input-format", type=str, choices=["text", "jsonl"], default=None,
                    help="Input format: 'text' for text file, 'jsonl' for JSONL (auto-detected if not specified)")
    ap.add_argument("--context", type=Path, default=None,
                    help="Optional context file (plain text) used for all utts (for text format)")
    ap.add_argument("--context-inline", type=str, default=None,
                    help="Optional context string used for all utts (overrides --context, for text format)")
    ap.add_argument("--round1-questions", type=Path, default=None,
                    help="Path to Round 1 clarification questions file (format: uttid question_text)")
    ap.add_argument("--round1-answers", type=Path, default=None,
                    help="Path to Round 1 user response answers file (format: uttid-round1\\tanswer_text)")
    ap.add_argument("--original-asr", type=Path, default=None,
                    help="Path to original ASR file (format: uttid asr_text) for Round 2+ context")
    ap.add_argument("--model", type=str, default="gpt-5.2-2025-12-11", 
                    help="Model name (default: gpt-5.2-2025-12-11)")
    
    args = ap.parse_args()
    
    # Auto-detect input format if not specified
    input_format = args.input_format
    if input_format is None:
        # Check file extension
        if args.input.suffix == ".jsonl":
            input_format = "jsonl"
        else:
            # Try to detect by reading first line
            try:
                with args.input.open("r", encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line.startswith("{") and "uttid" in first_line:
                        input_format = "jsonl"
                    else:
                        input_format = "text"
            except:
                input_format = "text"  # default to text
    
    if args.context_inline is not None:
        context_text = args.context_inline.strip()
    elif args.context is not None:
        context_text = args.context.read_text(encoding="utf-8").strip()
    else:
        context_text = ""  # empty for round 1 typically, or will be generated from JSONL for Round 2+
    
    # Load Round 1 QA files for Round 2+ context
    qa_map = None
    if args.round_num >= 2:
        if args.round1_questions is not None or args.round1_answers is not None:
            qa_map = load_round1_qa_files(args.round1_questions, args.round1_answers)
        else:
            print("[INFO] No Round 1 QA files provided, will use JSONL context or empty context", file=sys.stderr)
    
    # Load original ASR file if provided
    original_asr_map = None
    if args.original_asr is not None and args.original_asr.exists():
        original_asr_map = {}
        with args.original_asr.open("r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_line(line)
                if parsed is not None:
                    uttid, asr_text = parsed
                    original_asr_map[uttid] = asr_text
        print(f"[INFO] Loaded {len(original_asr_map)} original ASR entries", file=sys.stderr)
    
    print(f"[INFO] Model: {args.model}", file=sys.stderr)
    
    process_file(
        input_path=args.input,
        output_path=args.output,
        context_text=context_text,
        round_num=args.round_num,
        model=args.model,
        input_format=input_format,
        qa_map=qa_map,
        original_asr_map=original_asr_map,
    )


if __name__ == "__main__":
    main()

