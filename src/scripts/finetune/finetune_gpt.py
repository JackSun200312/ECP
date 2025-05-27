import json
from openai import OpenAI

import os
# Now you can make API calls

def fine_tune(fine_tuning_json_path, model = 'gpt-4o-2024-08-06'):
    client = OpenAI()
    
    # Upload the training file
    file_response = client.files.create(
        file=open(fine_tuning_json_path, "rb"),
        purpose="fine-tune"
    )
    file_id = file_response.id
    print(f"Uploaded file ID: {file_id}")
    
    # Create fine-tuning job
    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=model
    )
    
    job_id = fine_tune_response.id
    print(f"Fine-tuning job created: {job_id}")
    return job_id

# Example usage
fine_tuning_json_path = "Formalization/fine_tuning_data.jsonl"
fine_tune(fine_tuning_json_path)
