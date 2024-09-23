Preprocessing:
 use this link to trim a video:https://tools.invideo.io/tools/video-trimmer/

############
For YOLOv8x (the extra-large variant of YOLOv8), the default input images size should be 640x640 pixels.

############
use this like :https://github.com/HumanSignal/labelImg/releases to download LableImg
which helps to annotate the images which has availability to get YOLO annoatation marks

###########
dataset should be organized as follows:

custom_dataset/
│
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── test/  # Optional, for testing only
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   ├── val/
│   │   ├── img1.txt
│   │   ├── img2.txt
│   │   └── ...
│   └── test/  # Optional, for testing only
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
<<<<<<< pranith
######## To activate GPU ########
I use python 3.9 to fine-tune the yolo, so to utilize the GPU, install necessary libraries with corresponding versionb as mentioned below

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117


#####train YOLO with custom dataset####

For that run the ToTrainYOLO.py file by feeding below details
        
        model_path='yolov8x.pt',
        yaml_path=r"D:\AI Projects\fineTuneYolo8xModel\FineTuningYOLO\throwing_parcel.yaml",
        project_name='yolo_training',
        run_name='yolov8_finetuned',
        epochs=50,
        batch_size=4,
        image_size=640

=======
>>>>>>> main


