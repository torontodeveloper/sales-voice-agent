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

os.makedirs("data", exist_ok=True)
customer_positive_scenarios = [
    "Customer is excited to sign up for a green energy plan",
    "Customer wants to upgrade to a premium plan after seeing bill savings",
    "Customer calls to thank the agent and asks about referral program",
    "Customer is happy with service and wants to add another property",
    "Customer asks about loyalty rewards and long-term contract benefits",
    "Customer wants to switch to renewable energy and asks for best plan",
    "Customer is moving and wants to transfer service to new address",
    "Customer asks about budget billing to make payments predictable",
]
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
customer_scenarios = [*customer_positive_scenarios, *customer_complaint_scenarios]


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

base_dir = Path(__file__).parent.parent.parent
print(f'path is {base_dir}')
data_output_dir = base_dir/"data"/"processed"/"synthetic_sg_dataset.jsonl"
print(f'home path is {data_output_dir}')
# run this file first to create if not already there otherwise,it will overwite
if __name__ == "__main__":
    with open(data_output_dir, "w") as file:
        for scenario in customer_scenarios:
            print(f"scenari is {scenario}")
            for _ in range(30):
                print(f"iteration", _)
                conversation = generate_dialogue(scenario)
                conversation["id"] = f"{str(uuid.uuid4())}"
                file.write(json.dumps(conversation) + "\n")
                file.flush()
                time.sleep(2)
    print("Done!!!", flush=True)
