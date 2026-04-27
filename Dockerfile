# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System libraries ──────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0  → OpenCV
# ffmpeg                 → Whisper audio processing
# libgomp1               → PyTorch / EasyOCR parallel inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer cache) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy app source ───────────────────────────────────────────────────────────
COPY . .

# ── Pre-download the YOLOv8 nano model at build time ─────────────────────────
# This avoids a cold-start download on every container restart.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# ── Port ─────────────────────────────────────────────────────────────────────
# HF Spaces expects 7860. Render injects PORT env var — default to 7860.
EXPOSE 7860

# ── Start with Gunicorn (production WSGI server) ──────────────────────────────
# 1 worker to keep RAM low on free tiers; timeout 300s for slow model cold-starts
CMD sh -c "gunicorn --bind 0.0.0.0:${PORT:-7860} --workers 1 --timeout 300 app:app"
