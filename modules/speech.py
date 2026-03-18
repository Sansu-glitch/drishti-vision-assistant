import whisper
import sounddevice as sd
import numpy as np
import queue
import time
import os
from gtts import gTTS
import pygame
import tempfile

print("Loading speech model...")
model = whisper.load_model("base")
print("Whisper model loaded!")

q = queue.Queue()
pygame.mixer.init()

def speak(text, lang='en'):
    print("Assistant:", text)
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            temp_file = f.name
        tts.save(temp_file)
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        pygame.mixer.music.unload()
        os.remove(temp_file)
    except Exception as e:
        print("Speech error:", e)

def listen(timeout=5):
    print("Listening...")
    sample_rate = 16000
    duration = timeout

    # Record audio
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()

    # Convert to format Whisper expects
    audio = np.squeeze(audio)

    # Transcribe with Whisper
    result = model.transcribe(audio, fp16=False)
    text = result["text"].strip()
    detected_lang = result.get("language", "en")

    print(f"You said: {text} (Language: {detected_lang})")
    return text, detected_lang