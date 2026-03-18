from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

print("Loading scene model...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_scene(frame=None):
    if frame is None:
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return "I could not capture the scene."

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=60)
    description = processor.decode(output[0], skip_special_tokens=True)
    return description