from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import uuid
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI API Key is not set in the environment")

client = OpenAI(api_key=OPENAI_API_KEY)

objection_list = [
    "Your rates are too high",
    "I'm happy with my current provider",
    "I don't want a long-term contract",
    "I've heard bad things about ABC Energy",
    "I don't have time for this right now",
    "Can you just send me something in the mail",
    "I already switched last year",
    "Your competitor offered me a better deal",
    "I don't understand my current bill",
    "I'm renting, my landlord handles electricity",
]


def generate_dialogue(objection: str):
    prompt = f"""You are helping train a sales AI. 
    Customer said: "{objection}"                                                                                   
                    
    Generate a JSON with:                                                                                          
    - "chosen": excellent ABC Energy sales response (empathetic, addresses objection, moves toward close)
    - "rejected": poor response (gives up, rude, or unhelpful)                                                     
    
    Return only JSON: {{"chosen": "...", "rejected": "..."}}"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        text={"format": {"type": "json_object"}},
        timeout=30,
    )
    result = json.loads(response.output_text)
    print(f"response of gpt4 {result}")
    return {
        "id": str(uuid.uuid4()),
        "prompt": objection,
        "chosen": result["chosen"],
        "rejected": result["rejected"],
    }


base_dir = Path(__file__).parent.parent.parent
print(f'Base dire {base_dir}')
output_dir = base_dir/"data"/"processed"/"dpo_generated_pair.jsonl.jsonl"
if __name__ == "__main__":
    dpo_pair = []
    for objection_scenario in objection_list:
        for _ in range(10):
            response = generate_dialogue(objection_scenario)
            dpo_pair.append(response)

    with open(output_dir, "w") as file:
        for line in dpo_pair:
            file.write(json.dumps(line) + "\n")
