from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import numpy as np
import cv2
from modules.vision import detect_objects
from modules.ocr import read_text_from_camera
from modules.scene import describe_scene
from modules.currency import detect_currency

app = Flask(__name__)
CORS(app)

def decode_image(base64_image):
    img_data = base64.b64decode(base64_image)
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
