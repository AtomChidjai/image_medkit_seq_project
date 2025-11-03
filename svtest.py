import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

MODEL_PATH = 'multimed_2.pt'

model = YOLO(MODEL_PATH)
tracker = sv.ByteTrack()
label_annotator = sv.LabelAnnotator() 
box_annotator = sv.BoxAnnotator()

def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame, verbose=False)[0] 
    
    detections = sv.Detections.from_ultralytics(results)
    
    detections = tracker.update_with_detections(detections)
    
    labels = [
        f"{model.names[class_id]} {confidence:.2f}"
        for _, _, confidence, class_id, _
        in detections
    ]
    
    annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    return annotated_frame

CAMERA_SOURCE = 0

cap = cv2.VideoCapture(CAMERA_SOURCE)

if not cap.isOpened():
    print(f"Error: Could not open camera {CAMERA_SOURCE}.")
    exit()

print("Webcam feed running. Press 'q' to exit.")

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        annotated_frame = process_frame(frame)
        
        cv2.imshow("Real-Time Object Tracking", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Warning: Could not read frame from camera. Exiting.")
        break

cap.release()
cv2.destroyAllWindows()