import os, cv2

class Preprocessor:

    def __init__(self, config: dict):

        self.config = config

    def get_categories(self, path: str):
        """
        Reads and returns the names of folders in the specified path.

        Returns:
            list: A list of folder names within the specified path.
        """

        try:
            # List only directories
            folder_names = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder))]
            return folder_names
        
        except FileNotFoundError:
            print(f"Error: The path '{path}' does not exist.")
            return []
        
        except Exception as e:
            print(f"An error occurred: {e}")
            return []

        # Function to load images and labels
    def load_images_and_labels(self, path: str):
        # Image size for resizing
        img_size = (self.config["img_size"]["width"], self.config["img_size"]["height"])

        # Defining categories
        categories = self.get_categories(path)

        data = []
        labels = []

        for label, category in enumerate(categories):
            category_path = os.path.join(path, category)

            if not os.path.exists(category_path):
                print(f"Directory not found: {category_path}")
                continue

            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                try:
                    # Read image as grayscale
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"Could not read image: {img_path}")
                        continue

                    # Resize image
                    img = cv2.resize(img, img_size)

                    # Add to dataset
                    data.append(img.flatten())  # Flatten for DataFrame
                    labels.append(label)

                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        return data, labels
