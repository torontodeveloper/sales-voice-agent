import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import httpx

VLLM_URL = "https://gothic-dyslexic-overstate.ngrok-free.dev/v1/chat/completions"
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI API Key is not set in the environment")

client = OpenAI(api_key=OPENAI_API_KEY)


def evaluate_adverserial_llm_response(
    customer_message: str, agent_response: str
) -> json:
    prompt = f"""You are an expert sales coach evaluating an energy sales agent.
                                                                                                    
  Customer said: "{customer_message}"
  Agent responded: "{agent_response}"                                                               
                  
  Score the agent's response (1-10 each):
  1. Professionalism
  2. Empathy
  3. Objection handling
  4. Sales effectiveness
                                                                                                    
  Return JSON:
  {{"professionalism": X, "empathy": X, "objection_handling": X, "sales_effectiveness": X,          
  "overall": X, "feedback": "..."}}"""
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        text={"format": {"type": "json_object"}},
        timeout=30,
    )
    return json.loads(response.output_text)


customer_message = "My energy bill is too high, I want to cancel!"
agent_response = (
    "I understand your frustration. Let me look at your account and find a better plan."
)


def get_agent_response(customer_message: str) -> str:
    response = httpx.post(
        VLLM_URL,
        json={
            "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": "You are an ABC Energy sales agent."},
                {"role": "user", "content": customer_message},
            ],
            "max_tokens": 200,
        },
        headers={"ngrok-skip-browser-warning": "true"},
        timeout=30,
    )
    return response.json()["choices"][0]["message"]["content"]


def evaluate_test_set(test_file, output_file):
    results = []
    with open(test_file) as file:
        test_data = [json.loads(line) for line in file]

    # Filter energy records first
    energy_records = [r for r in test_data if "EnergyPlan" in r.get("services", [])]
    print(f"Total energy records in test set: {len(energy_records)}")

    for record in energy_records[]:
        if "EnergyPlan" not in record.get("services", []):
            continue
        convs = record["conversations"]
        # get first human turn and last gpt turn
        human = next(t["value"] for t in convs if t["from"] == "human")
        agent_response = get_agent_response(human)  # calls vLLM
        score = evaluate_adverserial_llm_response(human, agent_response)  # GPT judges
        score["id"] = record["id"]
        results.append(score)
        print(f"Evaluated {record['id']}: overall={score['overall']}", flush=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    evaluate_test_set("data/splits/test_data.jsonl", "data/gpt4llm_eval_results.json")
