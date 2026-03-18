import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'])

def detect_currency(frame=None):
    if frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "I could not capture the image."

    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray, detail=0)
    text = " ".join(result).lower()
    print("Text found on note:", text)

    denominations = ["2000", "500", "200", "100", "50", "20", "10"]
    for amount in denominations:
        if amount in text:
            return f"This is a {amount} rupees note."

    if "reserve bank" in text or "india" in text:
        return "This looks like an Indian currency note but I could not read the amount clearly."

    return "I could not identify the currency. Please hold the note closer."
