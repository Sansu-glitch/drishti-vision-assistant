from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
import os
import tempfile
from dotenv import load_dotenv

from modules.vision import detect_objects
from modules.ocr import read_text_from_camera
from modules.scene import describe_scene
from modules.currency import detect_currency

# Load .env from project root (works locally; on cloud, env vars are injected by platform)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)

# ── lazy-load Whisper ─────────────────────────────────────────────────────────
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
    np_arr   = np.frombuffer(img_data, np.uint8)
    frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

def text_to_audio(text, lang='en'):
    from gtts import gTTS
    tts   = gTTS(text=text, lang=lang, slow=False)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    return mp3_fp

def translate_to_hindi(text, action):
    hindi_templates = {
        'detect':   f"मैं देख सकता हूं: {text}",
        'read':     f"यहाँ लिखा है: {text}",
        'scene':    f"आपके आसपास: {text}",
        'currency': f"यह नोट है: {text}",
        'smart':    f"मैं देख सकता हूं: {text}",
    }
    return hindi_templates.get(action, text)

COMMAND_KEYWORDS = {
    'read':     ['read', 'padhho', 'padho', 'text', 'likha', 'likhna', 'ocr'],
    'currency': ['currency', 'note', 'rupee', 'paisa', 'rupaye', 'money', 'cash'],
    'scene':    ['scene', 'around', 'describe', 'surrounding', 'kya hai', 'dekho'],
    'detect':   ['detect', 'object', 'what', 'kya', 'saamne', 'identify'],
}

def map_command_to_action(command: str) -> str:
    cmd = command.lower()
    for action, keywords in COMMAND_KEYWORDS.items():
        if any(k in cmd for k in keywords):
            return action
    return 'detect'

# ── routes ────────────────────────────────────────────────────────────────────
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

@app.route('/speak', methods=['POST'])
def speak():
    try:
        data = request.json or {}
        text = data.get('text', '').strip()
        lang = data.get('lang', 'en')
        if not text:
            return jsonify({'result': 'No text provided', 'status': 'error'}), 400
        supported = {'en', 'hi', 'bn', 'ta', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'}
        if lang not in supported:
            lang = 'en'
        mp3_fp = text_to_audio(text, lang)
        return send_file(mp3_fp, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data       = request.json or {}
        frame      = decode_image(data['image'])
        question   = data.get('question', '').strip()
        scene_desc = describe_scene(frame)
        objects    = detect_objects(frame)
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

@app.route('/voice', methods=['POST'])
def voice():
    """Transcribe voice command and return the detected action.
    The frontend should then capture a camera frame and call the
    appropriate endpoint (/detect, /read, /scene, /currency) separately.
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'result': 'No audio file received', 'status': 'error'}), 400

        audio_file = request.files['audio']
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp:
            tmp_path = tmp.name
            audio_file.save(tmp_path)

        try:
            wmodel   = get_whisper()
            result   = wmodel.transcribe(tmp_path)
            command  = result.get('text', '').strip()
            language = result.get('language', 'en')
        finally:
            os.unlink(tmp_path)

        action = map_command_to_action(command)

        return jsonify({
            'command': command,
            'action': action,
            'language': language,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})

@app.route('/smart', methods=['POST'])
def smart():
    try:
        frame      = decode_image(request.json['image'])
        scene_desc = describe_scene(frame)
        objects    = detect_objects(frame)
        result     = f"{scene_desc}. Additionally, I can see: {objects}."
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


if __name__ == '__main__':
    # Local dev only — gunicorn is used in production (Docker)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)