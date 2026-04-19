from fastapi import FastAPI
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "sales-voice-agent"


app = FastAPI()


class ChatPrompt(BaseModel):
    system_prompt: str
    user_prompt: str


url = "https://gothic-dyslexic-overstate.ngrok-free.dev/v1/chat/completions"


@app.post("/call_sales_voice_agent")
def call_sales_voice_agent(chat_prompt: ChatPrompt):
    print(f"chat prompt {chat_prompt}")
    response = httpx.post(
        url,
        json={
            "model": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "messages": [
                {"role": "system", "content": f"{chat_prompt.system_prompt}"},
                {"role": "user", "content": f"{chat_prompt.user_prompt} "},
            ],
            "max_tokens": 200,
        },
    )
    print(
        f"response is {response.status_code}, {type(response)} {type(response.json())}"
    )
    response_data = response.json()
    print(f"response is {response_data["choices"][0]["message"]["content"]}")
    return response_data["choices"][0]["message"]["content"]
