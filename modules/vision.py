import cv2
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8n.pt")

def describe_scene(detections):
    if len(detections) == 0:
        return "I see nothing in front of you"
    counts = Counter(detections)
    parts = []
    for obj, count in counts.items():
        if count == 1:
            parts.append(f"one {obj}")
        else:
            parts.append(f"{count} {obj}s")
    return "I see " + " and ".join(parts)

def detect_objects(frame=None):
    if frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "Camera not working"
    frame = cv2.resize(frame, (320, 240))
    results = model(frame, verbose=False)
    objects = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            objects.append(name)
    return describe_scene(objects)