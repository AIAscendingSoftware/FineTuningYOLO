import os,shutil,random
from math import ceil

class YOLODatasetPreparation:
    def __init__(self, source_folder, train_folder, test_folder, val_folder, train_ratio=0.75, test_ratio=0.15, val_ratio=0.15):
        self.source_folder = source_folder
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.val_folder = val_folder
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        
        # Create directories if they don't exist
        os.makedirs(self.train_folder, exist_ok=True)
        os.makedirs(self.test_folder, exist_ok=True)
        os.makedirs(self.val_folder, exist_ok=True)

    def get_image_list(self):
        """Get list of images in the source folder"""
        return [f for f in os.listdir(self.source_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    def split_dataset(self):
        """Split the dataset into train, test, and val sets"""
        images = self.get_image_list()
        total_images = len(images)
        #To suffle all the images in the folder
        random.shuffle(images)

        # Calculate the number of images for each split
        train_count = ceil(self.train_ratio * total_images)
        test_count = ceil(self.test_ratio * total_images)
        val_count = total_images - train_count - test_count  # Remaining for validation

        train_images = images[:train_count]
        test_images = images[train_count:train_count + test_count]
        val_images = images[train_count + test_count:]

        return train_images, test_images, val_images

    def save_images(self, images, target_folder, prefix="img"):
        """Save images to the target folder with sequential numbering"""
        for idx, image_name in enumerate(images, 1):
            source_path = os.path.join(self.source_folder, image_name)
            new_image_name = f'{prefix}{idx}.jpg'
            target_path = os.path.join(target_folder, new_image_name)
            shutil.copy(source_path, target_path)

    def prepare(self):
        """Main process to split and save images"""
        # Split dataset into train, test, and val sets
        train_images, test_images, val_images = self.split_dataset()

        print(f"Total Images: {len(train_images) + len(test_images) + len(val_images)}")
        print(f"Train: {len(train_images)}, Test: {len(test_images)}, Val: {len(val_images)}")

        # Save images to respective folders with proper naming
        self.save_images(train_images, self.train_folder, prefix="img")
        self.save_images(test_images, self.test_folder, prefix="img")
        self.save_images(val_images, self.val_folder, prefix="img")

# Usage
if __name__ == "__main__":
    source_folder = r"E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\preprocessing\all_images"
    train_folder = r"E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\FineTuningYOLO\custom_dataset\images\train"
    test_folder = r"E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\FineTuningYOLO\custom_dataset\images\test"
    val_folder = r"E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\FineTuningYOLO\custom_dataset\images\val"

    # Instantiate the dataset preparation class and execute
    dataset_preparation = YOLODatasetPreparation(source_folder, train_folder, test_folder, val_folder)
    dataset_preparation.prepare()
