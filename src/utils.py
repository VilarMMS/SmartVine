import os, re, shutil
from math import ceil


class Utils:

    def __init__(self, config: dict):
    
        self.config = config

        if not os.path.exists("experiments"):
            os.makedirs("experiments")  # Create experiments directory if it doesn't exist

    def create_experiment_directory(self):

        """
        This method creates a directory for the experiment, using the 'config' passed during initialization.
        """
        # Get the base directory path using the model name
        name = self.config.general.name
        self.base_path = f"experiments/{name}"

        # Check if the directory already exists
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)  # Create the directory if it doesn't exist
        else:
            # Get a list of all directories in 'experiments' folder
            existing_dirs = [d for d in os.listdir("experiments") if d.startswith(name)]
            
            # Find the highest index of the existing experiment directories
            max_index = -1
            for d in existing_dirs:
                # Use regular expression to find any numbers at the end of the directory name
                match = re.search(r'(\d+)$', d)
                if match:
                    index = int(match.group(1))
                    max_index = max(max_index, index)
            
            # Update the name of the experiment to the next available index
            name = f"{self.config.general.name}{max_index + 1:01d}"
            
            # Set the new name in config
            self.config.general.name = name
            
            # Create the new directory
            os.makedirs(f"experiments/{name}")

        # with open(os.path.join("experiments",name,"config.json"),'w') as f:
        #     json.dump(self.config, f, indent = 4)
        
        print(f"Directory created: experiments/{self.config.general.name}")

    def split_dataset(self):

        train_dir = "data/train"
        test_dir = "data/test"

        # Iterate through each subdirectory in train
        for subdir in os.listdir(train_dir):
            train_subdir = os.path.join(train_dir, subdir)
            test_subdir = os.path.join(test_dir, subdir)
            
            # Skip if it's not a directory
            if not os.path.isdir(train_subdir):
                continue
            
            # Ensure corresponding test subdirectory exists
            os.makedirs(test_subdir, exist_ok=True)
            
            # List all files in the training subdirectory
            files = os.listdir(train_subdir)
            
            # Calculate the number of files to move
            num_files_to_move = ceil(len(files) * self.config["split_ratio"])
            
            # Select the files to move
            files_to_move = files[:num_files_to_move]
            
            # Move files to the test subdirectory
            for file in files_to_move:
                shutil.move(os.path.join(train_subdir, file), os.path.join(test_subdir, file))

        print("Split process complete")
