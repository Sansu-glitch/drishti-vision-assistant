# Drishti — AI Vision Assistant

An AI powered wearable vision assistant for visually impaired and elderly people in India.

## Problem
Over 12 million visually impaired people in India cannot read medicine prescriptions, identify currency, or understand their surroundings independently. Existing solutions like OrCam cost ₹2,50,000 — unaffordable for most Indians.

## Solution
Drishti is an affordable AI powered vision assistant that uses a camera and voice to help visually impaired people understand the world around them.

## Features
- Object Detection — identifies objects in front of user
- Text Reading (OCR) — reads medicine, signs, books
- Scene Description — describes full surroundings
- Currency Detection — identifies Indian rupee notes
- Multilingual Voice Commands — supports Hindi and English

## How to Run
pip install -r requirements.txt
python app.py
Open browser → http://127.0.0.1:5000

## Tech Stack
- YOLOv8 — object detection
- EasyOCR — text recognition
- BLIP — scene description
- Whisper — multilingual speech recognition
- Flask — web backend
- gTTS — text to speech
- OpenCV — image processing

## Target Users
- Visually impaired people
- Elderly people who cannot read small text
- People with low vision

## Future Plans
- Flutter mobile app
- ESP32-CAM glasses hardware
- Hindi, Bengali, Tamil language support
- Face recognition
- Medicine reminder system

## Project Domain
Artificial Intelligence and Assistive Technology

## Team
Sanskar Singh