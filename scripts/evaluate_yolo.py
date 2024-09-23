from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO("D:\AI Projects\fineTuneYolo8xModel\yolo_training\yolov8_finetuned5\weights\best.pt")

# Evaluate the model on the validation set
model.val(data='throwing_parcel.yaml')
