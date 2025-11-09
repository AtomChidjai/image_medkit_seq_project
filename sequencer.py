# sequencer.py

import cv2
import numpy as np
from ultralytics import YOLO
import logging
from io import StringIO # For in-memory logging
import time

class Sequencer:
    """
    Handles YOLO model initialization, object tracking, sequence verification,
    and session-based logging.
    """
    
    def __init__(self, model_path, tracking_list, conf=0.25, iou=0.7, imgsz=256, tolerance=10):
        # Configuration
        self.model_path = model_path
        self.tracking_list = tracking_list
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tolerance = tolerance
        self.session_id = time.strftime("%Y%m%d_%H%M%S") # Unique ID for this session/log

        # State Variables
        self.expected_index = 0
        self.processed_objects = {}
        self.message = "Ready"
        self.is_sequence_complete = False
        self.last_event_type = None # 'Correct', 'Wrong', 'Reset', or None

        # Logging Setup (In-Memory Buffer)
        self.log_stream = StringIO()
        self.logger = self._setup_logger()
        
        # Load Model
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Model loaded: {self.model_path}")
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {self.model_path}")
            self.model = None

    def _setup_logger(self):
        """Sets up a logger that outputs to the in-memory stream."""
        logger = logging.getLogger(f"Sequencer_{self.session_id}")
        logger.setLevel(logging.INFO)
        # Prevent log messages from propagating to the root logger (which might print to console)
        logger.propagate = False 

        # Handler to write to the in-memory StringIO buffer
        handler = logging.StreamHandler(self.log_stream)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Clear existing handlers if they exist (crucial for Streamlit)
        if logger.hasHandlers():
            logger.handlers.clear()
            
        logger.addHandler(handler)
        
        logger.info(f"--- STARTING NEW SEQUENCE: {', '.join(self.tracking_list)} ---")
        return logger

    def get_log_content(self):
        """Returns the entire content of the in-memory log buffer."""
        return self.log_stream.getvalue()

    def is_touching_border(self, boxA, boxB):
        # ... (Your existing is_touching_border function remains the same) ...
        x1, y1, x2, y2 = boxA
        bx1, by1, bx2, by2 = boxB
        tolerance = self.tolerance
    
        touch_left   = abs(x1 - bx1) <= tolerance and y2 > by1 and y1 < by2
        touch_right  = abs(x2 - bx2) <= tolerance and y2 > by1 and y1 < by2
        touch_top    = abs(y1 - by1) <= tolerance and x2 > bx1 and x1 < bx2
        touch_bottom = abs(y2 - by2) <= tolerance and x2 > bx1 and x1 < bx2
    
        return touch_left or touch_right or touch_top or touch_bottom

    def check_detection(self, detected_name):
        """Checks the detected object against the next expected item in the list."""
        alert_message = ""
        self.last_event_type = None
        
        if self.expected_index < len(self.tracking_list):
            expected_name = self.tracking_list[self.expected_index]
            if detected_name == expected_name:
                self.expected_index += 1
                self.last_event_type = 'Correct' # New state for app.py alert
                if self.expected_index < len(self.tracking_list):
                    alert_message = f"Correct: {detected_name}. Next: {self.tracking_list[self.expected_index]}"
                else:
                    alert_message = "All items detected in order!"
                    self.is_sequence_complete = True
            else:
                self.last_event_type = 'Wrong' # New state for app.py alert
                alert_message = f"Wrong order: expected {expected_name}, got {detected_name}"
        
        self.message = alert_message
        
    def reset_state(self):
        """Hard reset of tracking state."""
        self.expected_index = 0
        self.processed_objects.clear()
        self.is_sequence_complete = False
        self.message = f"Tracking reset. Expected: {self.tracking_list[0]}" if self.tracking_list else "Ready"
        self.logger.warning("Manual reset performed.")

    def process_frame(self, frame):
        """Processes a single frame for detection, tracking, and sequence verification."""
        self.last_event_type = None # Reset event type every frame

        if self.model is None:
            return frame, self.get_display_message()

        # Run tracking on the current frame
        # ... (rest of tracking call remains the same) ...
        results = self.model.track(
            source=frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, 
            verbose=False, persist=True, show=False, save=False
        )
        
        # Check for auto-reset condition (Cleared state)
        if self.is_sequence_complete and not self.processed_objects:
            self.reset_state()
            self.message = f"Sequence complete and all items cleared. Tracking soft reset. Expected: {self.tracking_list[self.expected_index]}"
            self.logger.info("Auto-reset triggered: Area clear after sequence completion.")

        if not results:
            return frame, self.get_display_message()
            
        r = results[0]
        annotated_frame = r.plot()
        # ... (frame, ROI drawing remains the same) ...
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
                        
                        # Log the event
                        self.logger.info(f"ID {track_id} ({class_name}) detected. Result: {self.message}")
                        
                        # Use print for console debugging only
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