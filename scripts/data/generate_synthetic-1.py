from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time
import uuid

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI API Key is not set in the environment")

client = OpenAI(api_key=OPENAI_API_KEY)

os.makedirs("data", exist_ok=True)

customer_complaint_scenarios = [
    "Customer says their energy bill is very high and threatens to cancel if costs aren't reduced",
    "Customer wants to switch from their current provider because of poor service",
    "Customer is frustrated with unexpected charges on their bill and demands explanation",
    "Customer wants to reduce their carbon footprint and asks about green energy plans",
    "Customer is moving to a new home and needs to set up energy service quickly",
    "Customer asks to compare two plans before committing to one",
    "Customer is on a fixed income and needs the most affordable plan available",
    "Customer had an outage and is angry, agent must retain them with a better plan",
]


def generate_dialogue(scenario: str):
    """
    Generate Sythetic conversation dataset in the form of ShareGPT using GPT-4o for cost reduction
    but can be replaced with gpt-5 or later
    """
    prompt = f"""Generate a realistic multi-turn sales conversation between a customer and an energy company sales 
  agent.                                                                                                                  
  Scenario: {scenario}
                                                                                                                          
  Return JSON in this exact format:
  {{
      "id": "synthetic_001",
      "services": ["EnergyPlan"],
      "conversations": [                                                                                                  
          {{"from": "system", "value": "You are a professional energy sales agent..."}},
          {{"from": "human", "value": "..."}},                                                                            
          {{"from": "gpt", "value": "..."}},
          ...                                                                                                             
      ]           
  }}
  Only return valid JSON, nothing else."""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        text={"format": {"type": "json_object"}},
        timeout=30,
    )
    return json.loads(response.output_text)


# run this file first to create if not already there otherwise,it will overwite
if __name__ == "__main__":
    with open("data/processed/sythetic_sgd_dataset.jsonl", "w") as file:
        for scenario in customer_complaint_scenarios:
            print(f"scenari is {scenario}")
            for _ in range(3):
                print(f"iteration", _)
                conversation = generate_dialogue(scenario)
                conversation["id"] = f"{str(uuid.uuid4())}"
                file.write(json.dumps(conversation) + "\n")
                file.flush()
                time.sleep(2)
    print("Done!!!", flush=True)
