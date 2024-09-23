import os
import cv2

# Input and output paths
video_folder = r'E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\FineTuningYOLO\preprocessing\videos'
output_folder = r'E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\FineTuningYOLO\preprocessing\all_images'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Unique image numbering across all videos
global_frame_count = 1

# Process each video in the folder
for video_file in os.listdir(video_folder):
    video_path = os.path.join(video_folder, video_file)
    
    # Check if the file is a video by checking its extension
    if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(video_path)
        
        # Ensure video file is opened correctly
        if not cap.isOpened():
            print(f"Failed to open video file: {video_file}")
            continue

        frame_count = 0

        # Loop through each frame of the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no more frames

            # Save the current frame as an image
            frame_filename = f'img{global_frame_count}.jpg'
            frame_filepath = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_filepath, frame)

            global_frame_count += 1
            frame_count += 1

        print(f"Extracted {frame_count} frames from {video_file}")
        cap.release()

print(f"Frames from all videos saved to {output_folder}")
