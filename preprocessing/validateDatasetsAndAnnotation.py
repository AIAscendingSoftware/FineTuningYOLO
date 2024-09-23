import os

class DatasetValidator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_extensions = ['.jpg', '.jpeg', '.png']
        self.label_extension = '.txt'

    def validate(self):
        for split in ['train', 'val', 'test']:
            image_folder = os.path.join(self.dataset_path, 'images', split)
            label_folder = os.path.join(self.dataset_path, 'labels', split)
            # print('image_folder:',image_folder)
            # print('label_folder:',label_folder)
            if not os.path.exists(image_folder) or not os.path.exists(label_folder):
                print(f"Folder missing: {split}")
                continue

            image_files = self.get_files(image_folder, self.image_extensions)
            # print('image_files:',image_files)
            label_files = self.get_files(label_folder, [self.label_extension])
            # print('label_files:',label_files)

            # Check for missing label files and unnecessary files
            self.validate_file_pairs(image_files, label_files, split)
    
    def get_files(self, folder, extensions):
        """ Get files in the folder with the given extensions """
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and self.is_valid_extension(f, extensions)]
        return files
    
    def is_valid_extension(self, filename, extensions):
        """ Check if the file has one of the valid extensions """
        return any(filename.lower().endswith(ext) for ext in extensions)

    def validate_file_pairs(self, image_files, label_files, split):
        """ Check image-label pairs and data inside label files """
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        # print('image_basenames:',image_basenames)
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        # print('label_basenames:',label_basenames)

        # Find missing labels for images
        for image in image_basenames:
            if image not in label_basenames:
                print(f"Missing label for image: {image}.jpg in {split} split.")

        # Find unnecessary label files
        for label in label_basenames:
            if label not in image_basenames:
                print(f"Unnecessary label file: {label}.txt in {split} split.")

        # Validate each label file
        self.validate_label_data(image_basenames, split)

    def validate_label_data(self, image_basenames, split):
        label_folder = os.path.join(self.dataset_path, 'labels', split)
        for image in image_basenames:
            label_file = os.path.join(label_folder, f"{image}.txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as file:
                    content = file.readlines()

                    if not content:
                        print(f"Empty label file: {image}.txt in {split} split.")
                    else:
                        for line_num, line in enumerate(content):
                            if not self.is_valid_label_format(line):
                                print(f"Improper data in {image}.txt on line {line_num + 1} in {split} split.")

    def is_valid_label_format(self, line):
        """ Check if the label format is valid (adjust this based on your requirements) """
        try:
            elements = list(map(float, line.strip().split()))
            if len(elements) < 5:
                return False  # YOLO format requires at least 5 elements: class_id, x, y, w, h
            return True
        except ValueError:
            return False


if __name__ == "__main__":
    # Example usage
    dataset_path = r'E:\AI Ascending Software\AS AI Projects\AI PackageGuard\AI-PackageGuard\yolo\FineTuningYOLO\custom_dataset'
    validator = DatasetValidator(dataset_path)
    validator.validate()
