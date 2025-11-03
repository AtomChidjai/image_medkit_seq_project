import os
import cv2
from ultralytics import YOLO

MODEL_PATH = 'multimed_2.pt'
SOURCE_INPUT = 0
OUTPUT_DIR = 'predict_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_object_detection():
    print(f"Loading model from: {MODEL_PATH}")
    
    try:
        model = YOLO(MODEL_PATH)

        print("Press 'q' or 'esc' on the detection window to stop.")
        
        results = model.track(
            source=SOURCE_INPUT,
            show=False,
            save=False,
            project=OUTPUT_DIR,
            name='tracking_run',
            conf=0.25,
            iou=0.7,
            imgsz=256,
            verbose=False,
            stream=True,
            tracker='bytetrack.yaml'
        )

        for r in results:
            frame = r.plot()
            height, width, _ = frame.shape
            margin = 40

            # Define detection box region
            detect_box = (margin, margin, width - margin, height - margin)  # (x1, y1, x2, y2)

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

            # Check for border collision
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    if is_touching_border((x1, y1, x2, y2), detect_box, tolerance=5):
                        print(f"⚠️ {class_name} collided with detection border!")

            cv2.imshow("EGCO486 PROJECT", frame)

            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
                break

        cv2.destroyAllWindows()

    except FileNotFoundError:
        print(f"File not found: {MODEL_PATH}")
    except Exception as e:
        print(e)


def is_touching_border(boxA, boxB, tolerance=5):
    """
    Check if boxA (object) touches or crosses the border of boxB (detection box).
    """
    x1, y1, x2, y2 = boxA
    bx1, by1, bx2, by2 = boxB

    # Touch/cross conditions: within a few pixels (tolerance) of any edge
    touch_left   = abs(x1 - bx1) <= tolerance and y2 > by1 and y1 < by2
    touch_right  = abs(x2 - bx2) <= tolerance and y2 > by1 and y1 < by2
    touch_top    = abs(y1 - by1) <= tolerance and x2 > bx1 and x1 < bx2
    touch_bottom = abs(y2 - by2) <= tolerance and x2 > bx1 and x1 < bx2

    return touch_left or touch_right or touch_top or touch_bottom


if __name__ == "__main__":
    run_object_detection()