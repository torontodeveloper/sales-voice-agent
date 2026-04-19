from locust import HttpUser, task
import json


# locust -f scripts/bench/locust_bench.py --host https://your-ngrok-url
class SalesVoiceAgent(HttpUser):
    @task
    def objection_handling(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are an ABC Energy sales agent."},
                    {
                        "role": "user",
                        "content": "Your rates are too high, I'm not interested",
                    },
                ],
                "max_tokens": 200,
            },
            headers={"Content-Type": "application/json"},
        )

    def new_customer(self):
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are an ABC Energy sales agent."},
                    {
                        "role": "user",
                        "content": "I want to switch my electricity provider",
                    },
                ],
                "max_tokens": 200,
            },
            headers={"Content-Type": "application/json"},
        )
