import cv2
from ultralytics import YOLO

class Camera:
    def __init__(self, model_path="multimed_2.pt", conf=0.25, imgsz=256):
        self.model = YOLO(model_path)
        self.conf = conf
        self.imgsz = imgsz

    def process(self, frame):
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)

        r = results[0]
        frame = r.plot()

        height, width, _ = frame.shape
        margin = 40
        cv2.rectangle(
            frame,
            (margin, margin),
            (width - margin, height - margin),
            (0, 255, 0),
            2
        )

        text = "Detection Box"
        cv2.putText(
            frame,
            text,
            (margin , margin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

        return frame