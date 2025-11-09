# sequencer.py

import cv2
import numpy as np
from ultralytics import YOLO

class Sequencer:
    """
    Handles YOLO model initialization, object tracking, and sequence verification logic.
    """
    
    def __init__(self, model_path, tracking_list, conf=0.25, iou=0.7, imgsz=256, tolerance=10):
        # Configuration
        self.model_path = model_path
        self.tracking_list = tracking_list
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tolerance = tolerance

        # State Variables
        self.expected_index = 0
        self.processed_objects = {}
        self.message = "Ready"
        self.is_sequence_complete = False
        
        # Load Model
        try:
            self.model = YOLO(self.model_path)
            print(f"Model loaded: {self.model_path}")
        except FileNotFoundError:
            print(f"ERROR: Model file not found at {self.model_path}")
            self.model = None

    def is_touching_border(self, boxA, boxB):
        """Checks if any part of boxA touches the border of boxB within tolerance."""
        x1, y1, x2, y2 = boxA
        bx1, by1, bx2, by2 = boxB
        tolerance = self.tolerance
    
        # Check for proximity to any of the four borders of boxB
        touch_left   = abs(x1 - bx1) <= tolerance and y2 > by1 and y1 < by2
        touch_right  = abs(x2 - bx2) <= tolerance and y2 > by1 and y1 < by2
        touch_top    = abs(y1 - by1) <= tolerance and x2 > bx1 and x1 < bx2
        touch_bottom = abs(y2 - by2) <= tolerance and x2 > bx1 and x1 < bx2
    
        return touch_left or touch_right or touch_top or touch_bottom

    def check_detection(self, detected_name):
        """Checks the detected object against the next expected item in the list."""
        alert_message = ""
        
        if self.expected_index < len(self.tracking_list):
            expected_name = self.tracking_list[self.expected_index]
            if detected_name == expected_name:
                self.expected_index += 1
                if self.expected_index < len(self.tracking_list):
                    alert_message = f"Correct: {detected_name}. Next: {self.tracking_list[self.expected_index]}"
                else:
                    alert_message = "All items detected in order!"
                    self.is_sequence_complete = True
            else:
                alert_message = f"Wrong order: expected {expected_name}, got {detected_name}"
        
        self.message = alert_message

    def reset_state(self):
        """Hard reset of tracking state."""
        self.expected_index = 0
        self.processed_objects.clear()
        self.is_sequence_complete = False
        self.message = f"Tracking reset. Expected: {self.tracking_list[0]}" if self.tracking_list else "Ready"

    def process_frame(self, frame):
        """Processes a single frame for detection, tracking, and sequence verification."""
        if self.model is None:
            return frame, self.get_display_message()

        # Run tracking on the current frame
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            persist=True,
            show=False,
            save=False
        )
        
        # Check for auto-reset condition (Cleared state)
        if self.is_sequence_complete and not self.processed_objects:
            self.reset_state()
            self.message = f"Sequence complete and all items cleared. Tracking soft reset. Expected: {self.tracking_list[self.expected_index]}"

        if not results:
            # If no detections, return frame as is
            return frame, self.get_display_message()
            
        r = results[0]
        annotated_frame = r.plot()
        height, width, _ = annotated_frame.shape
        margin = 20

        detect_box_xyxy = (margin, margin, width - margin, height - margin)
        
        # Draw Detection ROI
        cv2.rectangle(
            annotated_frame, (margin, margin), (width - margin, height - margin), (0, 255, 0), 2
        )
        cv2.putText(
            annotated_frame, f"ROI (Tol: {self.tolerance}px)", (margin, margin - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

        # Detection and tracking logic - SKIPPED if sequence is complete
        if not self.is_sequence_complete and r.boxes is not None and r.boxes.id is not None:
            current_track_ids = r.boxes.id.cpu().numpy().astype(int)
            
            for box, track_id, cls_id in zip(r.boxes.xyxy, current_track_ids, r.boxes.cls):
                box_xyxy = box.cpu().numpy().astype(int)
                class_name = self.model.names[int(cls_id)]
                
                if self.is_touching_border(box_xyxy, detect_box_xyxy):
                    if track_id not in self.processed_objects:
                        self.check_detection(class_name)
                        self.processed_objects[track_id] = class_name 
                        print(f"ID {track_id} ({class_name}) triggered. Result: {self.message}")

            # Object Cleanup Logic
            active_ids = set(current_track_ids)
            ids_to_remove = [tid for tid in self.processed_objects if tid not in active_ids]
            for tid in ids_to_remove:
                self.processed_objects.pop(tid, None)
        
        # Add display message to frame
        display_message = self.get_display_message()

        cv2.putText(
            annotated_frame, display_message, (10, height - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )

        return annotated_frame, display_message

    def get_display_message(self):
        """Generates the main status message for the UI."""
        if self.is_sequence_complete:
            return "SEQUENCE COMPLETE. Clear area to reset."
        elif self.expected_index < len(self.tracking_list):
            return f"Expected: {self.tracking_list[self.expected_index]}"
        else:
            return self.message