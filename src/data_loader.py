import pandas as pd
import numpy as np
import os, cv2

class DataLoader:

    def __init__(self, img_channels: int, target_size = None):
        self.img_channels = img_channels
        self.target_size = target_size

    def data_loader(self, base_path, batch_size = 32):
        """
        Load images in batches from subfolders of a given path into a single DataFrame.

        Args:
            base_path (str): Path containing subfolders with images.
            batch_size (int): Number of images to process per batch.

        Returns:
            pd.DataFrame: DataFrame containing all image data and labels.
        """
        data = []
        labels = []

        # Initialize an empty DataFrame to accumulate batches
        df_final = pd.DataFrame()

        # Traverse each folder in the base path
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)

            # Ensure it is a directory
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)

                    # Check if the file is a valid image
                    try:
                        if self.img_channels == 1:
                            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) / 255
                        else:
                            img = cv2.imread(file_path) / 255

                        if img is not None:  # Ensure the file is a valid image
                            if self.target_size:
                                img = cv2.resize(img, self.target_size)  # Resize image if target_size is specified

                            img_flattened = img.flatten()  # Flatten the image into a 1D array
                            data.append(img_flattened)
                            labels.append(folder_name)

                            # Process batch
                            if len(data) >= batch_size:
                                df_batch = pd.DataFrame(data)
                                df_batch['label'] = labels

                                # Append batch to the final DataFrame
                                df_final = pd.concat([df_final, df_batch], ignore_index=True)

                                # Clear memory for the next batch
                                data = []
                                labels = []

                    except Exception as e:
                        print(f"Skipping file {file_path}: {e}")

        # Process any remaining data
        if data:
            df_batch = pd.DataFrame(data)
            df_batch['label'] = labels
            df_final = pd.concat([df_final, df_batch], ignore_index=True)

        return df_final

    def load_train_and_test_raw_datasets(self, save=False):
        """
        Load raw datasets and optionally save them to CSV files.

        Args:
            save (bool): Whether to save the processed DataFrame to a CSV file.

        Returns:
            tuple: DataFrames for training and testing datasets.
        """
        path_train = "data/raw/train"
        path_test = "data/raw/test"

        df_train = self.data_loader(path_train)
        df_test = self.data_loader(path_test)

        if save:
            train_output = "data/split/train/train.csv"
            test_output = "data/split/test/test.csv"

            os.makedirs(os.path.dirname(train_output), exist_ok=True)
            df_train.to_csv(train_output, index=False)

            os.makedirs(os.path.dirname(test_output), exist_ok=True)
            df_test.to_csv(test_output, index=False)

        return df_train, df_test

    def load_split_datasets(self, name_dataset="dataset"):
        """
        Load pre-split datasets from saved CSV files.

        Args:
            name_dataset (str): Base name of the saved dataset.

        Returns:
            tuple: DataFrames for training and testing datasets.
        """
        df_train = pd.read_csv(f"data/split/train/{name_dataset}_train.csv")
        df_test = pd.read_csv(f"data/split/test/{name_dataset}_test.csv")
        return df_train, df_test