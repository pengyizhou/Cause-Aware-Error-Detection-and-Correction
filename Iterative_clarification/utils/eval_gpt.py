from tqdm import tqdm
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


def _process_single_judge_item(args):
    """Helper function to process a single judge item."""
    idx, question, reference, prediction, prompt_template, client = args
    
    evaluation_prompt = prompt_template.format(question=question, prediction=prediction, reference=reference)

    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=evaluation_prompt,
            reasoning={
                "effort": "none"
            }
        )
        
        output = response.output_text

    except Exception as e:
        print(f"Error in completion: {e}")
        sleep(1)
        output = "empty"

    
    # Map to scores
    try:
        rate_score = float(output.split()[-1])
        success = 1
    except:
        rate_score = 0.0
        success = 0

    sample_rating_detail = {
        'question'        : question,
        'reference'       : reference,
        'model_prediction': prediction,
        'judge_response'  : output,
        'rate_score'      : rate_score,
        'success'         : success,
    }

    return idx, sample_rating_detail


def gpt5_as_judge(model_path, input_data):
    """ Compute the score of the model on the given data."""

    client = OpenAI()

    # generation
    questions, references, predictions = input_data

    PROMPT_TEMPLATE = """\
            [Reference Answer]
            {reference}

            [Model Answer]
            {prediction}

            [Question]
            {question}

            [Task]
            Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details. If the model response is something like 'cannot decide', please rate as 0.
            Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
            Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
            Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
            Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
            Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
            Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
            Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
            Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

            Your response should be formatted as follows:
            Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
            Rating: (int)"""

    # Prepare arguments for parallel processing
    args_list = [
        (idx, question, reference, prediction, PROMPT_TEMPLATE, client)
        for idx, (question, reference, prediction) in enumerate(zip(questions, references, predictions))
    ]

    results_dict = {}
    
    # Process with 10 parallel workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_single_judge_item, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                idx, result = future.result()
                results_dict[idx] = result
            except Exception as e:
                print(f"Error processing item: {e}")
                # Find the index from the future if possible, otherwise use a default
                # For simplicity, we'll handle this in the reconstruction step
    
    # Reconstruct list in original order
    all_details = [results_dict.get(i, {
        'question': '',
        'reference': '',
        'model_prediction': '',
        'judge_response': 'empty',
        'rate_score': 0.0,
        'success': 0,
    }) for i in range(len(questions))]

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores) * 20
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}


    return judge_results, all_details




def _process_single_judge_binary_item(args):
    """Helper function to process a single binary judge item."""
    idx, question, reference, prediction, prompt_template, client = args
    
    evaluation_prompt = prompt_template.format(question=question, prediction=prediction, reference=reference)

    try:
        response = client.responses.create(
            model="gpt-5.2",
            input=evaluation_prompt,
            reasoning={
                "effort": "none"
            }
        )
        
        output = response.output_text

    except Exception as e:
        print(f"Error in completion: {e}")
        sleep(1)
        output = "empty"

    
    # Map to scores
    try:
        rate_score = float(output.split()[-1])
        success = 1
    except:
        rate_score = 0.0
        success = 0

    sample_rating_detail = {
        'question'        : question,
        'reference'       : reference,
        'model_prediction': prediction,
        'judge_response'  : output,
        'rate_score'      : rate_score,
        'success'         : success,
    }

    return idx, sample_rating_detail


def gpt5_as_judge_binary(model_path, input_data):
    """ Compute the score of the model on the given data."""

    client = OpenAI()

    # generation
    questions, references, predictions = input_data

    PROMPT_TEMPLATE = """\
            [Reference Answer]
            {reference}

            [Model Answer]
            {prediction}

            [Question]
            {question}

            [Task]
            Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
            Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance. Please give a score of 0 or 1. 
            Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
            Score0: The answer is wrong, providing incorrect or irrelevant information compared to the reference. 
            Score1: The answer is correct, capturing or covering the meaning from the reference.

            Your response should be formatted as follows:
            Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
            Rating: (int)"""

    # Prepare arguments for parallel processing
    args_list = [
        (idx, question, reference, prediction, PROMPT_TEMPLATE, client)
        for idx, (question, reference, prediction) in enumerate(zip(questions, references, predictions))
    ]

    results_dict = {}
    
    # Process with 10 parallel workers
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(_process_single_judge_binary_item, args) for args in args_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                idx, result = future.result()
                results_dict[idx] = result
            except Exception as e:
                print(f"Error processing item: {e}")
                # Find the index from the future if possible, otherwise use a default
                # For simplicity, we'll handle this in the reconstruction step
    
    # Reconstruct list in original order
    all_details = [results_dict.get(i, {
        'question': '',
        'reference': '',
        'model_prediction': '',
        'judge_response': 'empty',
        'rate_score': 0.0,
        'success': 0,
    }) for i in range(len(questions))]

    all_scores   = [detail['rate_score'] for detail in all_details]
    avg_score    = sum(all_scores) / len(all_scores) * 100
    success_rate = sum([detail['success'] for detail in all_details]) / len(all_details)

    judge_results = {'judge_score': avg_score, 'success_rate': success_rate}


    return judge_results, all_details