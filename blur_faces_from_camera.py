import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

model = YOLO("models/yolov8n-face.pt")
blur_annotator = sv.PixelateAnnotator(pixel_size=40)


while True:
    ret, frame = cap.read()

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    frame = blur_annotator.annotate(frame.copy(), detections=detections)

    cv2.imshow("Blurring faces", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
