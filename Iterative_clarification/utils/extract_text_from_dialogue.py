#!/usr/bin/env python3
import json
import os
import sys

input_file = sys.argv[1]
clarification_output = sys.argv[2]
fixed_asr_output = sys.argv[3]

with open(clarification_output, 'w', encoding='utf-8') as cf, \
     open(fixed_asr_output, 'w', encoding='utf-8') as af:
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse the main JSON object
            data = json.loads(line)
            
            # Extract custom_id
            custom_id = data.get('custom_id')
            if not custom_id:
                continue
            
            # Extract the text field from the nested structure
            try:
                text_content = data['response']['body']['output'][1]['content'][0]['text']
            except (KeyError, IndexError):
                print(f"Warning: Could not extract text for {custom_id}")
                continue
            
            # Parse the text field (which is a JSON string)
            try:
                parsed_text = json.loads(text_content)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse text JSON for {custom_id}")
                continue
            
            # Extract round, clarification_question, and fixed_asr
            round_num = parsed_text.get('round', 1)
            clarification_question = parsed_text.get('clarification_question', [])

            fixed_asr = parsed_text.get('fixed_asr', '')
            
            # Write to clarification_question file: id [question]
            if isinstance(clarification_question, list):
                # Join multiple questions if there are any
                question_text = ' '.join(clarification_question)
            else:
                question_text = str(clarification_question)
            
            cf.write(f"{custom_id} {question_text}\n")
            
            # Write to fixed_asr file: id [fixed_asr]
            af.write(f"{custom_id} {fixed_asr}\n")
            
            print(f"Processed {custom_id}")

print("Extraction complete!")
print(f"Created {clarification_output} and {fixed_asr_output}")

