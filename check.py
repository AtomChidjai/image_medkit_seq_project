from ultralytics import YOLO

model = YOLO('multimed_1.pt')
print(model.names)