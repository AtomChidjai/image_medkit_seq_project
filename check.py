from ultralytics import YOLO

model = YOLO('multimed_1.pt')
print(model.names)

import cv2

# Try to open and immediately release the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if cap.isOpened():
    cap.release()
    print("✅ Camera released successfully.")
else:
    print("⚠️ No active camera to release.")

cv2.destroyAllWindows()
