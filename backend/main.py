from fastapi import FastAPI, UploadFile, File
from speech_to_text import speech_to_text
from ai_chat import ask_ai
from text_to_speech import text_to_speech

app = FastAPI()

@app.post("/listen")
async def listen(audio: UploadFile = File(...)):
    audio_path = "input.wav"

    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    # Step 1: Voice → Text
    user_text = speech_to_text(audio_path)

    # Step 2: Text → AI Response
    ai_reply = ask_ai(user_text)

    # Step 3: Text → Voice
    text_to_speech(ai_reply)

    return {
        "heard": user_text,
        "answer": ai_reply
    }
