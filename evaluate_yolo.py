from ultralytics import YOLO

# Load the fine-tuned model
model = YOLO('runs/detect/train/weights/best.pt')

# Evaluate the model on the validation set
model.val(data='throwing_parcel.yaml')
