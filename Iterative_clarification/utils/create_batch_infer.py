#!/usr/bin/env python3

from openai import OpenAI
import time
import os
import sys
# set the universal API key

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
def upload_batch_file(file_path):
    client_file = OpenAI()
    response = client_file.files.create(
        file=open(file_path, "rb"),
        purpose="batch",
        expires_after={
            "anchor": "created_at",
            "seconds": 3600*25
        }
    )
    return response


def create_batch_inference(file_id):
    client_infer = OpenAI()
    response = client_infer.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h"
    )
    return response

def retrieve_batch_inference(batch_id):
    client_retrieve = OpenAI()
    response = client_retrieve.batches.retrieve(batch_id)
    return response


def retrieve_results_file(file_id):
    client_retrieve = OpenAI()
    response = client_retrieve.files.content(file_id)
    return response

def main():
    response = upload_batch_file(sys.argv[1])
    file_id = response.id
    
    print(f"Uploaded file with ID: {file_id}")
    response_batch = create_batch_inference(file_id)
    print(f"Created batch inference with ID: {response_batch.id}")

    while True:
        response_status = retrieve_batch_inference(response_batch.id)
        print(f"Batch inference status: {response_status.status}")

        if response_status.status == 'completed':
            print("Batch inference completed successfully.")
            break
        elif response_status.status == 'failed':
            print("Batch inference failed.")
            break
        # wait for 30 seconds
        time.sleep(30)
    if response_status.status == 'completed':
        try:
            response_results = retrieve_results_file(response_status.output_file_id)
            # save the result file
            with open(sys.argv[2], "w") as f:
                f.write(response_results.text)
        except ValueError:
            result_file = response_status.error_file_id
            response_results = retrieve_results_file(result_file)
            # save the result file
            with open("<PATH_TO_ERROR_LOG>", "w") as f:
                f.write(response_results.text)
    else:
        result_file = response_status.error_file_id
        response_results = retrieve_results_file(result_file)
        # save the result file
        with open("<PATH_TO_ERROR_LOG>", "w") as f:
            f.write(response_results.text)

if __name__ == "__main__":
    main()

