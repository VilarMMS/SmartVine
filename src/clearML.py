import os
from tensorflow.keras.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
from clearml import Task, Logger
import json
import numpy as np

class ClearMLConfig:

    def __init__(self, config_path: str):

        """
        Initialize a ClearML task for the project and task name.
        """
        with open(config_path,"r") as f:
            self.config = json.load(f)

        # Initialize the ClearML task
        self.task = Task.init(project_name=self.config["clearML"]["project_name"], task_name=self.config["clearML"]["task_name"])
        self.logger = Logger.current_logger()

    def log_hyperparams(self, hyperparams: dict, trial_number = None):
        """
        Log hyperparameters to ClearML.

        Args:
            hyperparams (dict): A dictionary of hyperparameters.
            trial_number (int or None): The trial number to uniquely identify the trial.
        """
        if self.task is None:
            raise RuntimeError("Task is not initialized. Call `initialize_task` first.")

        # Use trial_number to create a unique key for each trial
        trial_key = f"trial_{trial_number}" if trial_number is not None else "default_trial"
        
        # Retrieve the current hyperparameters logged under 'hyperparameters' (if any)
        existing_hyperparams = self.task.get_parameters_as_dict().get('hyperparameters', {})
        
        # Add or update the trial's hyperparameters
        updated_hyperparams = {**existing_hyperparams, trial_key: hyperparams}
        
        # Log the updated hyperparameters to ClearML
        self.task.connect({'hyperparameters': updated_hyperparams})

    def log_metrics_epoch(self, metrics: dict, epoch: int, decimals = 4):

        if epoch == 1:

            # Reset the learning curve by logging zeros for all epochs initially
            for i in range(0, self.config["model_params"]["epochs"] + 1):

                self.logger.report_scalar("Learning curve (epoch)", "Val Loss", iteration = i, value = np.nan)
                self.logger.report_scalar("Learning curve (epoch)", "Train Loss", iteration =i , value = np.nan)
        
        else:

            self.logger.report_scalar("Learning curve (epoch)", "Val Loss", iteration = epoch, value = round(metrics["val_loss"], decimals))
            self.logger.report_scalar("Learning curve (epoch)", "Train Loss", iteration = epoch, value = round(metrics["training_loss"], decimals))
        

    class ClearMLCallback(Callback):
        """
        Keras callback to log metrics to ClearML at the end of each epoch.
        """

        def __init__(self, clearml_config: 'ClearMLConfig'):
            self.clearml_config = clearml_config

        def on_epoch_end(self, epoch, logs = None):
            if logs is not None:
                # Log the metrics at the end of the epoch
                metrics = {
                    "val_loss": logs.get('val_loss'),
                    "training_loss": logs.get('loss')
                    }
                # Log metrics using the log_metrics_epoch method
                self.clearml_config.log_metrics_epoch(metrics, epoch)
