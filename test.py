import os
import cv2
from ultralytics import YOLO

MODEL_PATH = 'medx_mini.pt'
SOURCE_INPUT = 0
OUTPUT_DIR = 'predict_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_touching_border(boxA, boxB, tolerance=5):
    x1, y1, x2, y2 = boxA
    bx1, by1, bx2, by2 = boxB

    touch_left   = abs(x1 - bx1) <= tolerance and y2 > by1 and y1 < by2
    touch_right  = abs(x2 - bx2) <= tolerance and y2 > by1 and y1 < by2
    touch_top    = abs(y1 - by1) <= tolerance and x2 > bx1 and x1 < bx2
    touch_bottom = abs(y2 - by2) <= tolerance and x2 > bx1 and x1 < bx2

    return touch_left or touch_right or touch_top or touch_bottom

def check_detection(detected_name, tracking_list, expected_index):
    alert_message = ""
    if expected_index < len(tracking_list):
        expected_name = tracking_list[expected_index]
        if detected_name == expected_name:
            expected_index += 1
            if expected_index < len(tracking_list):
                alert_message = f"Correct: {detected_name}. Next: {tracking_list[expected_index]}"
            else:
                alert_message = "All items detected in order!"
        else:
            alert_message = f"Wrong order: expected {expected_name}, got {detected_name}"
    
    return expected_index, alert_message

def run_object_detection():
    tracking_list = ['Eno', 'Mybacin']
    expected_index = 0
    processed_objects = {} 
    message = '' 
    tolerance = 10
    
    is_sequence_complete = False 
    
    print(f"Loading model from: {MODEL_PATH}")

    try:
        model = YOLO(MODEL_PATH)
        
        cap = cv2.VideoCapture(SOURCE_INPUT, cv2.CAP_DSHOW)

        print("Press 'x' to reset tracking, or 'q'/'esc' to stop. Check 'detection_log.txt' for events.")
        
        while cap.isOpened():
            _, frame = cap.read()

            results = model.track(
                source=frame,
                show=False,
                save=False,
                conf=0.25,
                iou=0.7,
                imgsz=256,
                verbose=False,
                persist=True
            )

            if not results:
                # If no detections, skip drawing/processing logic but run cleanup check
                pass
            else:
                r = results[0]
                annotated_frame = r.plot()
                height, width, _ = annotated_frame.shape
                margin = 20
    
                detect_box_xyxy = (margin, margin, width - margin, height - margin)
    
                # Draw Detection ROI (Green box)
                cv2.rectangle(
                    annotated_frame, (margin, margin), (width - margin, height - margin), (0, 255, 0), 2
                )
                cv2.putText(
                    annotated_frame, f"Detection ROI (Border Trigger, Tol: {tolerance}px)", (margin, margin - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                )
    
                # Object detection and tracking logic - SKIPPED if sequence is complete
                if not is_sequence_complete and r.boxes is not None and r.boxes.id is not None:
                    current_track_ids = r.boxes.id.cpu().numpy().astype(int)
                    
                    for box, track_id, cls_id in zip(r.boxes.xyxy, current_track_ids, r.boxes.cls):
                        box_xyxy = box.cpu().numpy().astype(int)
                        class_name = model.names[int(cls_id)]
                        
                        if is_touching_border(box_xyxy, detect_box_xyxy, tolerance):
                            if track_id not in processed_objects:
                                expected_index, message = check_detection(class_name, tracking_list, expected_index)
                                processed_objects[track_id] = class_name 
                                
                                print(f"ID {track_id} ({class_name}) triggered border proximity. Result: {message}")

                                # Check if this was the final correct item
                                if expected_index == len(tracking_list):
                                    is_sequence_complete = True
                
            # Object Cleanup Logic (MUST run every frame)
            active_ids = set(r.boxes.id.cpu().numpy().astype(int)) if r.boxes is not None and r.boxes.id is not None else set()
            ids_to_remove = [tid for tid in processed_objects if tid not in active_ids]
            for tid in ids_to_remove:
                processed_objects.pop(tid, None)
            
            # Auto-reset check after completion
            if is_sequence_complete and not processed_objects:
                expected_index = 0
                is_sequence_complete = False
                message = f"Sequence complete and all items cleared. Tracking soft reset. Expected: {tracking_list[expected_index]}"
                print(message)
                
            # Display current expected item/status
            if is_sequence_complete:
                display_message = "SEQUENCE COMPLETE. Clear area to reset."
            elif expected_index < len(tracking_list):
                display_message = f"Expected: {tracking_list[expected_index]}"
            else:
                display_message = message 

            # Only display frame if results were processed (r is available)
            if 'r' in locals():
                cv2.putText(
                    annotated_frame, display_message, (10, height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
                )
    
                cv2.imshow("EGCO486 PROJECT", annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in [ord('x')]:
                expected_index = 0
                processed_objects.clear()
                is_sequence_complete = False
                message = f"Tracking reset. Expected: {tracking_list[expected_index]}"
                print(message)
            elif key in [ord('q'), 27]:
                break

        cap.release()
        cv2.destroyAllWindows()

    except FileNotFoundError:
        error_msg = f"File not found: {MODEL_PATH}"
        print(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred: {e}"
        print(error_msg)

if __name__ == "__main__":
    run_object_detection()