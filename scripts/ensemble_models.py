

from ultralytics import YOLO
import cv2
import numpy as np

class YOLOEnsemble:
    def __init__(self, model_paths):
        """
        Initialize the YOLO ensemble with given model paths.
        
        :param model_paths: List of paths to the pre-trained model files.
        """
        self.models = [YOLO(path) for path in model_paths]
        print("Models loaded successfully.")
    
    def run_inference(self, source):
        """
        Run inference on a video using all models in the ensemble and display the results in real-time.
        
        :param source: Path to the video for inference.
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"Error opening video file: {source}")
            return

        print("Video file opened successfully.")

        # Define colors for different classes
        np.random.seed(42)  # Seed to keep color assignments consistent across runs
        num_classes = 80  # Adjust this number if needed (depends on the number of classes in your model)
        colors = np.random.randint(0, 255, size=(num_classes, 3), dtype='uint8')  # Random colors for each class
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            combined_frame = frame.copy()

            for model in self.models:
                results = model.predict(source=frame, save=False, conf=0.25)  # Adjust confidence threshold if needed
                
                for result in results:
                    boxes = result.boxes  # Get detected boxes
                    
                    # Draw bounding boxes and labels
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Extract bounding box coordinates
                        conf = box.conf.cpu().numpy()[0]  # Confidence score
                        cls = int(box.cls.cpu().numpy()[0])  # Class ID
                        label = f"{model.names[cls]} {conf:.2f}"  # Class label and confidence
                        
                        # Assign color based on class ID
                        color = [int(c) for c in colors[cls % len(colors)]]
                        
                        # Draw the rectangle and label on the frame
                        cv2.rectangle(combined_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(combined_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the combined frame with detections
            cv2.imshow("YOLO Ensemble", combined_frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == '__main__':
    model_paths = [
        r'D:\AI Projects\fineTuneYolo8xModel\models\fine_tuned_models\weights\best.pt',  # Fine-tuned model
        r'D:\AI Projects\fineTuneYolo8xModel\yolov8x.pt'  # Pre-trained model
    ]
    
    # Create an instance of the YOLOEnsemble class with both models
    yolo_ensemble = YOLOEnsemble(model_paths)
    
    # Run inference on a video
    yolo_ensemble.run_inference(r'D:\AI Projects\fineTuneYolo8xModel\throwingParcelsVideos\WhatsApp Video 2024-08-30 at 21.13.02.mp4')
