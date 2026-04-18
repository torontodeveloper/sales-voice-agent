import datasets
import json
import os
from dotenv import load_dotenv
import pprint


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
    with open("data/processed/sythetic_sgd_dataset.jsonl", "a+") as file:
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
