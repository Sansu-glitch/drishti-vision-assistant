from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

READ_KEYWORDS = ["read", "text", "what does", "written", "says", "medicine",
                 "prescription", "letter", "word", "पढ़", "पढो"]
DETECT_KEYWORDS = ["detect", "object", "front", "see", "what is", "around",
                   "surroundings", "who", "person", "what are", "सामने", "क्या है"]
SCENE_KEYWORDS = ["describe", "scene", "look", "tell me", "explain",
                  "what is happening", "doing", "what can you see", "बताओ", "देखो"]
CURRENCY_KEYWORDS = ["currency", "note", "money", "rupee", "cash", "how much",
                     "denomination", "नोट", "पैसे", "रुपये"]
EXIT_KEYWORDS = ["exit", "bye", "goodbye", "stop", "quit", "बंद", "रुको"]


def understand_command_fallback(text):
    text_lower = text.lower()
    if any(k in text_lower for k in READ_KEYWORDS):
        return "read"
    if any(k in text_lower for k in DETECT_KEYWORDS):
        return "detect"
    if any(k in text_lower for k in SCENE_KEYWORDS):
        return "scene"
    if any(k in text_lower for k in CURRENCY_KEYWORDS):
        return "currency"
    if any(k in text_lower for k in EXIT_KEYWORDS):
        return "exit"
    return "unknown"


def understand_command(text):
    prompt = f"""
You are a command classifier for a vision assistant for visually impaired people.
The user will speak a command in English or Hindi or any Indian language.
You must classify it into exactly one of these actions:

- read → user wants to read text (medicine, signs, books)
- detect → user wants to know what objects are in front
- scene → user wants a full description of surroundings
- currency → user wants to identify money/notes
- exit → user wants to stop the assistant
- unknown → cannot classify

User said: "{text}"

Reply with ONLY one word from the list above. Nothing else.
"""
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        action = response.text.strip().lower()
        valid = ["read", "detect", "scene", "currency", "exit", "unknown"]
        return action if action in valid else "unknown"
    except Exception as e:
        print(f"[Gemini error] {e} — falling back to keyword matching.")
        return understand_command_fallback(text)