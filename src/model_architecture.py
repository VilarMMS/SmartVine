import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.Utils.clearML import ClearMLConfig as ClearML
import pandas as pd
import numpy as np
import os

class Architecture(tf.keras.Model):

    def __init__(self, config: dict):
        
        super().__init__()

        # Model configuration
        self.config = config
        self.clearml = clearML


    