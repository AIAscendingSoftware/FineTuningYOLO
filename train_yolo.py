from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8x.pt')  # Using the yolov8x model

# Fine-tune the model on the custom dataset
model.train(data='E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\yolo_objects_detection\fineTuneYOLO\throwing_parcel.yaml', epochs=50, imgsz=640, batch=16)

'''This script will automatically save the 
best model weights (best.pt) after training.'''