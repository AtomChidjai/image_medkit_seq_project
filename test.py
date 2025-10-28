import os
from ultralytics import YOLO

# --- Configuration ---
# 1. Path to your custom .pt file (kept your change)
CUSTOM_MODEL_PATH = 'med_1.pt' 

# 2. Source for inference. 
# Changed to '0' to use the default webcam.
SOURCE_INPUT = 0 

# 3. Define where any saved output would go (kept your change)
OUTPUT_DIR = 'predict_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------

def run_object_detection():
    """
    Loads an Ultralytics YOLO model from a .pt file and performs 
    real-time object tracking on a live camera feed using ByteTrack.
    """
    print(f"Loading model from: {CUSTOM_MODEL_PATH}")
    
    try:
        # Load the custom model.
        model = YOLO(CUSTOM_MODEL_PATH)
        
        # --- Run Tracking on Camera ---
        print("\nStarting live tracking feed... Press 'q' or 'esc' on the detection window to stop.")
        print("Objects will now have persistent IDs thanks to ByteTrack.")
        
        # We use model.track() instead of model.predict() for object tracking.
        # 'tracker' specifies the algorithm to use (ByteTrack is fast and reliable).
        # 'imgsz=480' keeps the resolution low for better FPS on your PC.
        results = model.track(
            source=SOURCE_INPUT,
            show=True,              # <<< Displays the video stream with bounding boxes and IDs
            save=False,
            project=OUTPUT_DIR,
            name='tracking_run',    # New run name for tracking
            conf=0.25,
            iou=0.7,
            imgsz=480,              # Lower resolution for better FPS
            verbose=False,
            stream=True,            # Use streaming for efficient video/camera processing
            tracker='bytetrack.yaml' # <<< ENABLED TRACKING HERE
        )
        
        # This loop runs continuously and prints frame-level information
        for r in results:
            # Check if tracking data is available and print the number of tracked objects
            if r.boxes and r.boxes.id is not None:
                 print(f"Frame processed. Tracked objects: {len(r.boxes.id)}", end='\r', flush=True)
            else:
                 print(f"Frame processed. Detections found: {len(r.boxes)}", end='\r', flush=True)

        print("\n--- Tracking Stream Closed ---")

    except FileNotFoundError:
        print(f"Error: Model file not found at {CUSTOM_MODEL_PATH}. Please check the path.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during tracking. Check if the camera ({SOURCE_INPUT}) is available.")
        print(f"Details: {e}")

if __name__ == "__main__":
    # NOTE: Ensure you have a webcam connected and the 'ultralytics' library installed.
    run_object_detection()
