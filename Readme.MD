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



