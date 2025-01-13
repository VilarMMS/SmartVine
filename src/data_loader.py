import pandas as pd
import numpy as np
import os, cv2

class DataLoader:

    def __init__(self, gray_scale = True, target_size = None):

        self.gray_scale = gray_scale
        self.target_size = target_size

    def data_loader(self, base_path):

        """
        Load images from subfolders of a given path into a pandas DataFrame.
        Each image is flattened into a row, and the folder name is added as a 'label' column.

        Args:
            base_path (str): Path containing subfolders with images.
            target_size (tuple): Desired size (width, height) to resize the images. Default is None (no resizing).

        Returns:
            pd.DataFrame: DataFrame containing image data and labels.
        """
        data = []
        labels = []

        # Traverse each folder in the base path
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            
            # Ensure it is a directory
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    
                    # Check if the file is a valid image
                    try:
                        if self.gray_scale:
                            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)/255
                        else:
                            img = cv2.imread(file_path)/255

                        if img is not None:  # Ensure the file is a valid image
                            if self.target_size:
                                img = cv2.resize(img, self.target_size)  # Resize image if target_size is specified

                            img_flattened = img.flatten()  # Flatten the image into a 1D array
                            data.append(img_flattened)
                            labels.append(folder_name)
                    except Exception as e:
                        print(f"Skipping file {file_path}: {e}")

        # Create a DataFrame
        df = pd.DataFrame(data)
        df['label'] = labels

        return df
    
    def load_train_and_test_raw_datasets(self, save: False):

        path_train = "data/raw/train"
        path_test = "data/raw/test"

        df_train = DataLoader.data_loader(path_train)
        df_test = DataLoader.data_loader(path_test)

        if save:
            path_split_train = "data/split/train/"
            
            if os.path.isdir(path_split_train):
                pass
            else:
                os.makedirs(path_split_train, exist_ok = True)

            df_train.to_csv(f"{path_split_train}_train.csv", index = False)

            path_split_test = "data/split/test/"
            if os.path.isdir(path_split_test):
                pass
            else:
                os.makedirs(path_split_test, exist_ok = True)

            df_train.to_csv(f"{path_split_test}_test.csv", index = False)

        return df_train, df_test
    
    def load_split_datasets(self):

        df_train = pd.read(f"data/split/train/{self.name_dataset}_train.csv")
        df_test = pd.read(f"data/split/train/{self.name_dataset}_test.csv")

        return df_train, df_test



