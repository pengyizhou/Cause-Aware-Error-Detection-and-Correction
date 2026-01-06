#!/usr/bin/env python3
"""
dialogue_gpt.py

This script reads questions from a text file (format: uttid question),
sends each question to GPT-5.2, and writes the answers to result.txt.
"""

import argparse
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI


def parse_text_file(text_path):
    """Parse the text file and extract (uttid, question) pairs."""
    entries = []
    with open(text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Split on first space: uttid is the first token, rest is the question
            parts = line.split(' ', 1)
            if len(parts) == 2:
                uttid, question = parts
                entries.append((uttid, question))
    return entries


def _process_single_question(args):
    """Process a single question and return the answer."""
    idx, uttid, question, client = args

    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=question,
            reasoning={
                "effort": "none"
            }
        )
        
        answer = response.output_text
        # Clean answer: remove newlines for single-line output
        answer = ' '.join(answer.split())

    except Exception as e:
        print(f"Error processing {uttid}: {e}")
        sleep(1)
        answer = "Error: Unable to generate response."

    return idx, uttid, answer


def run_dialogue_gpt(text_path, output_path, max_workers=10):
    """Run GPT-5.2 on all questions and save results."""
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Parse input file
    entries = parse_text_file(text_path)
    print(f"Loaded {len(entries)} questions from {text_path}")
    
    # Prepare arguments for parallel processing
    args_list = [
        (idx, uttid, question, client)
        for idx, (uttid, question) in enumerate(entries)
    ]
    
    results_dict = {}
    
    # Process with parallel workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process_single_question, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing questions"):
            try:
                idx, uttid, answer = future.result()
                results_dict[idx] = (uttid, answer)
            except Exception as e:
                print(f"Error processing item: {e}")
    
    # Reconstruct list in original order and write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(len(entries)):
            if i in results_dict:
                uttid, answer = results_dict[i]
            else:
                uttid = entries[i][0]
                answer = "Error: Processing failed."
            f.write(f"{uttid} {answer}\n")
    
    print(f"Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Query GPT-5.2 with questions and save answers")
    parser.add_argument("--text", type=str, required=True, 
                        help="Path to the text file containing questions (format: uttid question)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output result.txt file")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of parallel workers (default: 10)")
    
    args = parser.parse_args()
    
    run_dialogue_gpt(args.text, args.output, args.workers)


if __name__ == "__main__":
    main()

