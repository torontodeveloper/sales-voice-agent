import datasets
import json
import os
from dotenv import load_dotenv
import pprint
from pathlib import Path


# run this file after synthetic file
def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError("HuggingFace Access Token is not set in the environment")

    share_gpt_data = datasets.load_dataset(
        "Mediform/sgd-sharegpt",
        "function_cot_nlg",
        trust_remote_code=True,
        token=HF_TOKEN,
    )

    sample = share_gpt_data["train"][0]
    pprint.pprint(sample.keys(), indent=4)
    pprint.pprint(sample["conversations"], indent=4)
    for item in sample["conversations"]:
        pprint.pprint(f'{item["from"]}============>, {item["value"]}')
    base_dir = Path(__file__).parent.parent.parent
    print(f'Base dire {base_dir}')
    output_dir = base_dir/"data"/"processed"/"synthetic_sg_dataset.jsonl"
    with open(output_dir, "a+") as file:
        print(f"Each row in SGD dataset ===> {share_gpt_data["train"][0]}")
        for row in share_gpt_data["train"]:
            record = {
                "id": row["dialogue_id"],
                "services": row["services"],
                "conversations": row["conversations"],
            }
            file.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
