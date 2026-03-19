import openai
import os

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

client = openai.OpenAI(api_key=API_KEY)

conversation = [
    {
        "role": "system",
        "content": "You are a helpful, intelligent female voice assistant."
    }
]

def ask_ai(user_text: str) -> str:
    conversation.append({"role": "user", "content": user_text})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation
    )

    reply = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": reply})

    return reply
