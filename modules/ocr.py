import easyocr
import cv2
import numpy as np

reader = easyocr.Reader(['en'])

def read_text_from_camera(frame=None):
    if frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "I could not capture image"

    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    result = reader.readtext(gray, detail=1)
    text_list = [res[1] for res in result if res[2] > 0.4]
    return " ".join(text_list) if text_list else "No text found"