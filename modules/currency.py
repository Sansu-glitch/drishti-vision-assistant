import cv2
import numpy as np
import easyocr

reader = easyocr.Reader(['en'])

# ── Color-based heuristics for Indian currency notes ─────────────────────────
# Each note has a dominant color family. We check HSV ranges.
# These are approximate — good as a fallback when OCR fails.
COLOR_HINTS = [
    # (denomination_string, hue_low, hue_high, sat_low, val_low, min_pixels)
    ('2000', 140, 170, 40, 80, 4000),   # magenta / pink
    ('500',  100, 140, 30, 80, 4000),   # stone-lavender / soft blue
    ('200',   15,  30, 60, 80, 4000),   # bright orange
    ('100',   90, 115, 40, 80, 4000),   # blue-green / lavender
    ('50',    35,  65, 40, 80, 4000),   # fluorescent green
    ('20',    20,  35, 60, 80, 4000),   # yellow-green
    ('10',   100, 130, 20, 80, 3000),   # chocolate / dark teal
]

def detect_by_color(frame):
    """Returns a denomination string if color matches, else None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for denom, h_lo, h_hi, s_lo, v_lo, min_px in COLOR_HINTS:
        mask = cv2.inRange(hsv,
                           np.array([h_lo, s_lo, v_lo]),
                           np.array([h_hi, 255, 255]))
        if cv2.countNonZero(mask) >= min_px:
            return denom
    return None

# ── Main detection function ───────────────────────────────────────────────────
def detect_currency(frame=None):
    if frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "I could not capture the image."

    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # ── Step 1: OCR-based detection (most reliable when print is clear) ──────
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = reader.readtext(gray, detail=0)
    text   = " ".join(result).lower()
    print("Currency OCR text:", text)

    denominations = ["2000", "500", "200", "100", "50", "20", "10"]
    for amount in denominations:
        if amount in text:
            return f"This is a {amount} rupees note."

    # ── Step 2: Color-based fallback ─────────────────────────────────────────
    color_match = detect_by_color(frame)
    if color_match:
        return f"This looks like a {color_match} rupees note (identified by color)."

    # ── Step 3: Generic Indian currency check ────────────────────────────────
    if any(kw in text for kw in ["reserve bank", "india", "bharat", "satyamev"]):
        return "This looks like an Indian currency note, but I could not read the amount clearly. Please hold it closer and ensure the number is visible."

    return "I could not identify the currency. Please hold the note closer with the denomination clearly visible."