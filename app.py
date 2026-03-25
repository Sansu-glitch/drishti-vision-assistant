from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
import os
import tempfile

from modules.vision import detect_objects
from modules.ocr import read_text_from_camera
from modules.scene import describe_scene
from modules.currency import detect_currency

app = Flask(__name__)
CORS(app)

# ── lazy-load heavy models only when first used ──────────────────────────────
_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model

# ── helpers ───────────────────────────────────────────────────────────────────
def decode_image(base64_image):
    img_data = base64.b64decode(base64_image)
    np_arr  = np.frombuffer(img_data, np.uint8)
    frame   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def text_to_audio(text, lang='en'):
    """Convert text to an in-memory MP3 using gTTS and return a BytesIO object."""
    from gtts import gTTS
    tts = gTTS(text=text, lang=lang, slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

COMMAND_KEYWORDS = {
    'read':     ['read', 'padhho', 'padho', 'text', 'likha', 'likhna', 'OCR'],
    'currency': ['currency', 'note', 'rupee', 'paisa', 'rupaye', 'money', 'cash'],
    'scene':    ['scene', 'around', 'describe', 'surrounding', 'kya hai', 'dekho'],
    'detect':   ['detect', 'object', 'what', 'kya', 'saamne', 'identify'],
}

def map_command_to_action(command: str) -> str:
    cmd = command.lower()
    for action, keywords in COMMAND_KEYWORDS.items():
        if any(k in cmd for k in keywords):
            return action
    return 'detect'   # sensible default

# ── existing routes ───────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

@app.route('/detect', methods=['POST'])
def detect():
    try:
        frame  = decode_image(request.json['image'])
        result = detect_objects(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

@app.route('/read', methods=['POST'])
def read_text():
    try:
        frame  = decode_image(request.json['image'])
        result = read_text_from_camera(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

@app.route('/scene', methods=['POST'])
def scene():
    try:
        frame  = decode_image(request.json['image'])
        result = describe_scene(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

@app.route('/currency', methods=['POST'])
def currency():
    try:
        frame  = decode_image(request.json['image'])
        result = detect_currency(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

# ── NEW: /speak — converts text to speech and returns an MP3 ─────────────────
@app.route('/speak', methods=['POST'])
def speak():
    """
    POST body: { "text": "...", "lang": "en" }   (lang is optional, defaults to 'en')
    Returns:   audio/mpeg  (MP3 bytes)
    The frontend plays this directly with new Audio(URL.createObjectURL(blob)).
    """
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        lang = data.get('lang', 'en')

        if not text:
            return jsonify({'result': 'No text provided', 'status': 'error'}), 400

        # gTTS only supports 'hi' for Hindi, 'en' for English
        # Anything else falls back to English
        supported = {'en', 'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'}
        if lang not in supported:
            lang = 'en'

        mp3_fp = text_to_audio(text, lang)
        return send_file(mp3_fp, mimetype='audio/mpeg')

    except Exception as e:
        # If gTTS fails (e.g. no internet), return error JSON so the
        # frontend falls back to the browser's built-in SpeechSynthesis
        return jsonify({'result': str(e), 'status': 'error'}), 500

# ── NEW: /ask — answers a free-form question about the current camera frame ──
@app.route('/ask', methods=['POST'])
def ask():
    """
    POST body: { "image": "<base64>", "question": "What colour is my shirt?" }
    Returns:   { "result": "...", "status": "success" }

    Strategy:
      1. Get a scene description from BLIP.
      2. Get object list from YOLO.
      3. Combine both into a plain-English answer that addresses the question.
    """
    try:
        data     = request.json or {}
        frame    = decode_image(data['image'])
        question = data.get('question', '').strip()

        # Run both modules
        scene_desc = describe_scene(frame)
        objects    = detect_objects(frame)

        # Build a combined answer
        if question:
            answer = (
                f"You asked: {question}. "
                f"Here is what I see: {scene_desc}. "
                f"Objects detected: {objects}."
            )
        else:
            answer = f"{scene_desc}. {objects}."

        return jsonify({'result': answer, 'status': 'success'})

    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

# ── NEW: /voice — accepts a WebM/OGG audio blob, transcribes with Whisper ────
@app.route('/voice', methods=['POST'])
def voice():
    """
    POST body: multipart/form-data with field 'audio' (WebM blob from MediaRecorder).
    Returns:
      {
        "command":  "padhho mujhe",   ← raw transcript
        "action":   "read",            ← mapped route name
        "language": "hi",              ← detected language
        "status":   "success"
      }

    The frontend should then call the mapped route automatically.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'result': 'No audio file received', 'status': 'error'}), 400

        audio_file = request.files['audio']

        # Save to a temp file (Whisper needs a real file path)
        suffix = '.webm'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        try:
            model   = get_whisper()
            result  = model.transcribe(tmp_path)
            command  = result.get('text', '').strip()
            language = result.get('language', 'en')
        finally:
            os.unlink(tmp_path)   # always clean up

        action = map_command_to_action(command)

        return jsonify({
            'command':  command,
            'action':   action,
            'language': language,
            'status':   'success'
        })

    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

# ── NEW: /smart — runs detect + scene together and returns a combined summary ─
@app.route('/smart', methods=['POST'])
def smart():
    """
    POST body: { "image": "<base64>" }
    Runs YOLO object detection AND BLIP scene description, returns both merged.
    """
    try:
        frame      = decode_image(request.json['image'])
        scene_desc = describe_scene(frame)
        objects    = detect_objects(frame)
        result     = f"{scene_desc}. Additionally, I can see: {objects}."
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)