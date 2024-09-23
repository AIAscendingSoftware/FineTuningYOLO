
import os
import torch
import psutil
import GPUtil
from ultralytics import YOLO

class YOLOTrainer:
    def __init__(self, model_path, yaml_path, project_name, run_name, epochs=50, batch_size=4, image_size=640):
        self.model_path = model_path
        self.yaml_path = yaml_path
        self.project_name = project_name
        self.run_name = run_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.use_gpu = self.check_gpu()
        self.model = YOLO(self.model_path)

    def check_gpu(self):
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]
            print(f"GPU is available: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Initial GPU memory usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            return True
        else:
            print("GPU is not available. Using CPU.")
            return False

    def print_resource_usage(self):
        cpu_percent = psutil.cpu_percent()
        ram_percent = psutil.virtual_memory().percent
        if self.use_gpu:
            gpu = GPUtil.getGPUs()[0]
            gpu_percent = gpu.memoryUsed / gpu.memoryTotal * 100
            print(f"CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}% | GPU Memory: {gpu_percent:.1f}%")
        else:
            print(f"CPU: {cpu_percent:.1f}% | RAM: {ram_percent:.1f}%")

    def validate_yaml_path(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"The YAML file does not exist at: {self.yaml_path}")

    def train(self):
        self.validate_yaml_path()

        # Train the model
        results = self.model.train(
            data=self.yaml_path,
            epochs=self.epochs,
            imgsz=self.image_size,
            batch=self.batch_size,
            name=self.run_name,
            project=self.project_name,
            device=0 if self.use_gpu else 'cpu',
            half=True  # Enable mixed precision
        )

        self.print_training_summary(results)

    def print_training_summary(self, results):
        print("Training complete. Best model weights saved as 'best.pt' in the run folder.")
        
        if results.maps:
            print("\nTraining Results:")
            print(f"mAP50-95: {results.maps[0]:.3f}")
            if len(results.maps) > 1:
                print(f"mAP50: {results.maps[1]:.3f}")
            else:
                print("Warning: Only one map value available, skipping mAP50.")
        else:
            print("Warning: No mAP results available.")

        # Safely check precision and recall metrics
        precision = results.results_dict.get('metrics/precision(B)', None)
        recall = results.results_dict.get('metrics/recall(B)', None)

        if precision is not None:
            print(f"Final epoch precision: {precision:.3f}")
        else:
            print("Warning: Precision metric not available.")

        if recall is not None:
            print(f"Final epoch recall: {recall:.3f}")
        else:
            print("Warning: Recall metric not available.")

        # Final GPU usage if applicable
        if self.use_gpu:
            gpu = GPUtil.getGPUs()[0]
            print(f"Final GPU memory usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")

if __name__ == '__main__':
    # Initialize the trainer with necessary parameters
    trainer = YOLOTrainer(
        model_path='yolov8x.pt',
        yaml_path=r"D:\AI Projects\fineTuneYolo8xModel\FineTuningYOLO\throwing_parcel.yaml",
        project_name='yolo_training',
        run_name='yolov8_finetuned',
        epochs=50,
        batch_size=4,
        image_size=640
    )

    # Start training the model
    trainer.train()
