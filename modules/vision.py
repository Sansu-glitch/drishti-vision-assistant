import cv2
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8n.pt")

def get_position(box, frame_width, frame_height):
    """Get horizontal position and distance estimate from bounding box."""
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    
    box_width  = x2 - x1
    box_height = y2 - y1
    center_x   = (x1 + x2) / 2
    box_area   = (box_width * box_height) / (frame_width * frame_height)

    # Horizontal position
    left_zone  = frame_width * 0.35
    right_zone = frame_width * 0.65

    if center_x < left_zone:
        position = "on your LEFT"
    elif center_x > right_zone:
        position = "on your RIGHT"
    else:
        position = "in front of you"

    # Distance estimate based on bounding box size
    if box_area > 0.35:
        distance = "very close"
    elif box_area > 0.15:
        distance = "close"
    elif box_area > 0.05:
        distance = "nearby"
    else:
        distance = "far away"

    return position, distance


def detect_objects(frame=None):
    if frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Camera not working"

    frame = cv2.resize(frame, (640, 480))
    frame_height, frame_width = frame.shape[:2]

    results = model(frame, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            cls        = int(box.cls[0])
            confidence = float(box.conf[0])
            name       = model.names[cls]

            # Only include confident detections
            if confidence < 0.45:
                continue

            position, distance = get_position(box, frame_width, frame_height)
            detections.append((name, position, distance, confidence))

    if not detections:
        return "I don't see any objects clearly in front of you."

    # Sort by confidence — most confident first
    detections.sort(key=lambda x: x[3], reverse=True)

    # Limit to top 4 objects to avoid overwhelming the user
    detections = detections[:4]

    parts = []
    for name, position, distance, _ in detections:
        parts.append(f"{name} {position}, {distance}")

    if len(parts) == 1:
        return f"I can see a {parts[0]}."
    else:
        result = "I can see: "
        result += "; ".join(parts[:-1])
        result += f"; and {parts[-1]}."
        return result


