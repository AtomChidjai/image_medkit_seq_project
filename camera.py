import cv2
from ultralytics import YOLO

class Camera:
    def __init__(self, model_path="multimed_2.pt", conf=0.25, imgsz=256, tracker="bytetrack.yaml"):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz
        self.tracker = tracker

    def process(self, frame):
        # Use model.track() instead of predict() to enable ByteTrack
        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            tracker=self.tracker,
            persist=True  # keep track IDs between frames
        )

        # There will be one result for this frame
        r = results[0]
        frame = r.plot()

        height, width, _ = frame.shape
        margin = 40

        # Draw detection box
        cv2.rectangle(
            frame,
            (margin, margin),
            (width - margin, height - margin),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Detection Box",
            (margin, margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        # Example: print active tracked IDs and classes
        if r.boxes.id is not None:
            for box in r.boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                print(f"Track ID {track_id} ({class_name}) detected")

        return frame
