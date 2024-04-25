import numpy as np
import supervision as sv
from ultralytics import YOLO

model = YOLO("models/yolov8m-seg.pt")
tracker = sv.ByteTrack()
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()


def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = list()
    for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
        labels.append(f"#{tracker_id} {results.names[class_id]}")

    annotated_frame = mask_annotator.annotate(frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels
    )


sv.process_video(
    source_path="cup.webm", target_path="results/cup.mp4", callback=callback
)
