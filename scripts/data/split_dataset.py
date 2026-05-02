import json
import random
import os
from pathlib import Path


def load_json_data(file_path):
    """
    Load JSON format line by line
    """
    with open(file_path, "r") as file:
        result = [json.loads(row) for row in file]
        print(f"row record is ====> {result[0]}")
        return result


def save_json_data(file_path, data):
    with open(file_path, "w") as file:
        for record in data:
            file.write(json.dumps(record) + "\n")

base_dir = Path(__file__).parent.parent.parent
output_dir = base_dir/"data"/"processed"/"synthetic_sg_dataset.jsonl"

shared_gpt_data = load_json_data(output_dir)
# shuffle the data in place
random.shuffle(shared_gpt_data)

total_records = len(shared_gpt_data)
print(f'Total Number of records {total_records}')
# Lets split the total dataset into 80/10/10 => train/validation/test
# This can be hyperparameter and can be experiemented
eighty_percent = int(0.8 * total_records)
ninety_perecent = int(0.9 * total_records)
shared_gpt_train_data = shared_gpt_data[:eighty_percent]
shared_gpt_validation_data = shared_gpt_data[eighty_percent:ninety_perecent]
shared_gpt_test_data = shared_gpt_data[ninety_perecent:]

save_json_data( base_dir/"data"/"processed"/"training_data.jsonl", shared_gpt_train_data)
save_json_data( base_dir/"data"/"processed"/"validation_data.jsonl", shared_gpt_validation_data)
save_json_data( base_dir/"data"/"processed"/"test_data.jsonl", shared_gpt_test_data)

# For, Debugging, TO-DO, will be deleted before deployed to production or before merge to main branch
print(f"Total Number of rows in synthetic and sgd data {total_records}")
print(f"Total Number of rows in Train Data=====> {len(shared_gpt_train_data)}")
print(
    f"Total Number of rows in Validation Data=====> {len(shared_gpt_validation_data)}"
)
print(f"Total Number of rows in Test Data =====>{len(shared_gpt_test_data)}")
