from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
import numpy as np
import cv2
import io
from gtts import gTTS
from google import genai
import os
from dotenv import load_dotenv
from modules.vision import detect_objects
from modules.ocr import read_text_from_camera
from modules.scene import describe_scene
from modules.currency import detect_currency

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or "AIzaSyDuTPRrYP4bpUK42alfCEgDuMG4v_a9yoA"
gemini_client = genai.Client(api_key=api_key)

app = Flask(__name__)
CORS(app)


def decode_image(base64_image):
    img_data = base64.b64decode(base64_image)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image — invalid or corrupt image data.")
    return frame


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = request.json
        frame = decode_image(data['image'])
        result = detect_objects(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


@app.route('/read', methods=['POST'])
def read_text():
    try:
        data = request.json
        frame = decode_image(data['image'])
        result = read_text_from_camera(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


@app.route('/scene', methods=['POST'])
def scene():
    try:
        data = request.json
        frame = decode_image(data['image'])
        result = describe_scene(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


@app.route('/currency', methods=['POST'])
def currency():
    try:
        data = request.json
        frame = decode_image(data['image'])
        result = detect_currency(frame)
        return jsonify({'result': result, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


@app.route('/speak', methods=['POST'])
def speak_text():
    try:
        data = request.json
        text = data.get('text', '')
        lang = data.get('lang', 'en')
        tts = gTTS(text=text, lang=lang)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return send_file(audio_fp, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', 'What do you see in this image?')
        frame_b64 = data.get('image', '')

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {
                    "parts": [
                        {"text": f"You are a helpful vision assistant for visually impaired people. Answer clearly and concisely. Question: {question}"},
                        {"inline_data": {"mime_type": "image/jpeg", "data": frame_b64}}
                    ]
                }
            ]
        )
        return jsonify({'result': response.text.strip(), 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


# ✅ Single /health route — duplicate removed
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

import requests as req

@app.route('/demo-image', methods=['POST'])
def demo_image():
    try:
        data = request.json
        url = data.get('url', '')
        response = req.get(url, timeout=10)
        img_data = base64.b64encode(response.content).decode('utf-8')
        return jsonify({'image': img_data, 'status': 'success'})
    except Exception as e:
        return jsonify({'result': str(e), 'status': 'error'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)