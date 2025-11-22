import cv2
import numpy as np
from ultralytics import YOLO
import logging
from io import StringIO
import time
from enum import Enum
from typing import Tuple, Optional, List, Set

class SequenceState(Enum):
    PREPARING = "PREPARING"
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    VALIDATING = "VALIDATING"
    COMPLETED = "COMPLETED"

class Sequencer:
    VALIDATION_DISPLAY_DURATION = 3.0
    
    def __init__(
        self,
        model_path: str, 
        tracking_list: List[str], 
        conf: float = 0.25, 
        iou: float = 0.7, 
        imgsz: int = 256, 
        tolerance: int = 10
    ):
        # Config
        self.model_path = model_path
        self.tracking_list = tracking_list
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tolerance = tolerance
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        
        # State machine
        self.current_state = SequenceState.PREPARING
        self.expected_index = 0
        
        # Tracking state
        self.classes_on_border_prev: Set[str] = set()
        self.classes_in_frame: Set[str] = set()
        
        # Validation & messaging
        self.message = "Initializing..."
        self.last_event_type: Optional[str] = None
        self.validation_start_time = 0.0
        
        # Logging setup
        self.log_stream = StringIO()
        self.logger = self._setup_logger()
        
        # Load model
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Model loaded: {self.model_path}")
            self._transition_to(SequenceState.PREPARING, "System initialized")
        except FileNotFoundError:
            self.logger.error(f"Model file not found at {self.model_path}")
            self.model = None
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Sequencer_{self.session_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        
        if logger.hasHandlers():
            logger.handlers.clear()
        
        handler = logging.StreamHandler(self.log_stream)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.info(f"--- STARTING NEW SEQUENCE: {', '.join(self.tracking_list)} ---")
        return logger
    
    def get_log_content(self) -> str:
        return self.log_stream.getvalue()
    
    def _transition_to(self, new_state: SequenceState, reason: str = ""):
        if self.current_state != new_state:
            old_state = self.current_state
            self.current_state = new_state
            log_msg = f"State transition: {old_state.value} → {new_state.value}"
            if reason:
                log_msg += f" ({reason})"
            self.logger.info(log_msg)
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        # l a b
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to l
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return blurred
    
    def _is_colliding_with_border(self, object_box: Tuple[int, int, int, int], border_box: Tuple[int, int, int, int]) -> bool:
        x1, y1, x2, y2 = object_box
        bx1, by1, bx2, by2 = border_box
        tol = self.tolerance
        
        # 1. (x1,y1) top left (x2,y2) bottom right with tol
        completely_inside = (
            x1 > bx1 + tol and 
            x2 < bx2 - tol and 
            y1 > by1 + tol and 
            y2 < by2 - tol
        )
        if completely_inside:
            return False
            
        # 2. check overlap
        overlap_x = max(0, min(x2, bx2) - max(x1, bx1))
        overlap_y = max(0, min(y2, by2) - max(y1, by1))
        has_overlap = overlap_x > 0 and overlap_y > 0
        if not has_overlap:
            return False
            
        # 3. check if touching any edge
        touch_left = abs(x1 - bx1) <= tol
        touch_right = abs(x2 - bx2) <= tol
        touch_top = abs(y1 - by1) <= tol
        touch_bottom = abs(y2 - by2) <= tol
        
        return touch_left or touch_right or touch_top or touch_bottom

    def reset_state(self):
        self.expected_index = 0
        self.classes_on_border_prev.clear()
        self.validation_start_time = 0.0
        self.last_event_type = 'Reset'
        
        self._transition_to(SequenceState.PREPARING, "Manual reset")
        self.logger.warning("Manual reset performed")
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        self.last_event_type = None
        
        if self.model is None:
            return frame, "Error: Model not loaded"
        
        ## preprocess l a b gaussian blur
        preprocessed = self._preprocess_frame(frame)
        results = self.model(
            source=preprocessed,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            show=False,
            save=False
        )
        
        if not results:
            return frame, self.get_display_message()
            
        r = results[0]
        annotated_frame = r.plot()
        
        height, width, _ = annotated_frame.shape
        margin = 20
        detect_box_xyxy = (margin, margin, width - margin, height - margin)
        
        # detection box
        cv2.rectangle(annotated_frame, (margin, margin), (width - margin, height - margin), (0, 255, 0), 2)
        
        # logic start here
        current_time = time.time()
        classes_on_border_curr = set()
        self.classes_in_frame = set()
        
        # analyze detections
        if r.boxes is not None:
            for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                box_xyxy = box.cpu().numpy().astype(int)
                class_name = self.model.names[int(cls_id)]
                
                self.classes_in_frame.add(class_name)
                
                # collide with detection box
                if self._is_colliding_with_border(box_xyxy, detect_box_xyxy):
                    classes_on_border_curr.add(class_name)
        
        # init state
        if self.current_state == SequenceState.PREPARING:
            # check all item in tracking list
            required_items = set(self.tracking_list)
            missing_items = required_items - self.classes_in_frame
            
            # if all item -> idle
            if not missing_items:
                self._transition_to(SequenceState.IDLE, "All items present")
                self.message = f"Ready! Start with: {self.tracking_list[0]}"
            else:
                self.message = f"Waiting for items: {', '.join(missing_items)}"
        
        # state completed -> prepare
        elif self.current_state == SequenceState.COMPLETED:
            self._transition_to(SequenceState.PREPARING, "Sequence finished, resetting")
            
        # active states (IDLE, TRACKING, VALIDATING)
        else:
            new_crossings = classes_on_border_curr - self.classes_on_border_prev
            
            for class_name in new_crossings:
                # check sequence
                if self.expected_index < len(self.tracking_list):
                    expected_class = self.tracking_list[self.expected_index]
                    
                    if class_name == expected_class:
                        # correct
                        self.expected_index += 1
                        self.last_event_type = 'Correct'
                        self.logger.info(f"✓ CORRECT: {class_name} leaving")
                        
                        if self.current_state == SequenceState.IDLE:
                            self._transition_to(SequenceState.TRACKING, "First item leaving")
                        
                        # check completion
                        if self.expected_index >= len(self.tracking_list):
                            self._transition_to(SequenceState.COMPLETED, "Sequence done")
                            self.logger.info("SEQUENCE COMPLETE!")
                        else:
                            next_item = self.tracking_list[self.expected_index]
                            self.message = f"✓ Correct: {class_name}. Next: {next_item}"
                            
                    else:
                        # wrong
                        self.last_event_type = 'Wrong'
                        self.validation_start_time = current_time
                        self.message = f"⚠ Warning: expected {expected_class}, got {class_name}"
                        self._transition_to(SequenceState.VALIDATING, f"Wrong item: {class_name}")
                        self.logger.warning(f"✗ WRONG: Expected {expected_class}, got {class_name}")

        # validation timeout
        if self.current_state == SequenceState.VALIDATING:
            if current_time - self.validation_start_time > self.VALIDATION_DISPLAY_DURATION:
                prev_state = SequenceState.TRACKING if self.expected_index > 0 else SequenceState.IDLE
                self._transition_to(prev_state, "Validation ended")

        # update state for next frame
        self.classes_on_border_prev = classes_on_border_curr
        
        # display message
        display_message = self.get_display_message()
        self._draw_status(annotated_frame, display_message, height)
        
        return annotated_frame, display_message

    def _draw_status(self, frame, message, height):
        color_map = {
            SequenceState.PREPARING: (0, 255, 255),
            SequenceState.IDLE: (200, 200, 200),
            SequenceState.TRACKING: (0, 255, 0),
            SequenceState.VALIDATING: (0, 0, 255),
            SequenceState.COMPLETED: (255, 165, 0)
        }
        color = color_map.get(self.current_state, (255, 255, 255))
        
        cv2.putText(frame, message, (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def get_display_message(self) -> str:
        if self.current_state == SequenceState.PREPARING:
            return self.message
        elif self.current_state == SequenceState.COMPLETED:
            return "✓ DONE! Put items back to reset."
        elif self.current_state == SequenceState.VALIDATING:
            return self.message
        elif self.current_state == SequenceState.TRACKING:
            return f"Progress: {self.expected_index}/{len(self.tracking_list)} | Next: {self.tracking_list[self.expected_index]}"
        else:
            if self.tracking_list:
                return f"Ready. Next: {self.tracking_list[0]}"
            return "Ready"

    def get_state_info(self) -> dict:
        return {
            'state': self.current_state.value,
            'progress': f"{self.expected_index}/{len(self.tracking_list)}",
            'active_objects': len(self.classes_in_frame),
            'message': self.message,
            'is_validating': self.current_state == SequenceState.VALIDATING,
            'validation_time_remaining': 0
        }