import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from src.clearML import ClearMLConfig as ClearML
import pandas as pd
import numpy as np
import os

class EfficientNet(tf.keras.Model):

    def __init__(self, config: dict, num_classes: int, clearML: object):
        super().__init__()
        """
        Initialize the EfficientNet model with hyperparameters from the config dictionary.

        Args:
            config (dict): Dictionary containing the model configuration.
                - input_shape (tuple): Shape of the input tensor (height, width, channels).
                - num_filters (int): Base number of channels for the convolution layers.
                - expansion_factor (int): Expansion factor for the MBConv blocks.
                - se_ratio (float): Squeeze-and-Excite ratio.
        """
        # Passinf the model configuration
        self.config = config

        # Passing the number of classes
        self.num_classes = num_classes

        # Unpacking optmiser settings
        self.optmiser = config["optimiser"]

        # Unpacking callback settings
        self.reduce_plateau = config["reduce_plateau"]

        # Input shape
        self.input_shape  =  (config['image_size']["width"], config['image_size']["height"], config['img_channels'])
        
        # Unpacking model parameters
        self.model_params = config["model_params"]
        
        # Passing ClearML object
        self.clearML = clearML

    def conv_block(self, x, filters, kernel_size, strides, activation = 'swish', use_bn = True):
        
        x  =  layers.Conv2D(filters, kernel_size, strides = strides, padding = "same", use_bias = not use_bn)(x)
        
        if use_bn:
            x  =  layers.BatchNormalization()(x)

        x  =  layers.Activation(activation)(x)

        return x

    def squeeze_and_excite(self, x):

        se_ratio  =  self.model_params['se_ratio']

        filters  =  x.shape[-1]
        reduced_filters  =  max(1, int(filters*se_ratio))

        se  =  layers.GlobalAveragePooling2D()(x)
        se  =  layers.Dense(reduced_filters, activation = "relu")(se)
        se  =  layers.Dense(filters, activation = "sigmoid")(se)

        return layers.Multiply()([x, se])

    def mb_conv_block(self, x, filters: int, kernel_size: int, kernel_init: str, strides: int):

        expansion_factor = self.model_params["expansion_factor"]
        input_channels  =  x.shape[-1]
        expanded_channels  =  input_channels*expansion_factor

        # Expansion phase
        if expansion_factor !=  1:
            x  =  self.conv_block(x, expanded_channels, kernel_size = 1)

        # Depthwise convolution
        x  =  layers.DepthwiseConv2D(kernel_size, 
                                     strides = strides, 
                                     padding = "same", 
                                     use_bias = False,
                                     kernel_init = kernel_init)(x)
        x  =  layers.BatchNormalization()(x)
        x  =  layers.Activation("swish")(x)

        # Squeeze and Excite
        x  =  self.squeeze_and_excite(x)

        # Output phase
        x  =  self.conv_block(x, filters, kernel_size = 1, activation = None, use_bn = True)

        return x

    def build_model(self):
        
        num_filters  =  self.model_params['num_filters']
        inputs  =  layers.Input(shape = self.input_shape)

        # Unpacking block parameters
        stem_params = self.config["model_params"]["stem"]
        core_params = self.config["model_params"]["core"]
        head_params = self.config["model_params"]["head"]

        # Stem
        x  =  self.conv_block(inputs, stem_params["num_filters"], stem_params["kernel_size"], stem_params["strides"])

        # Blocks
        for i in range(core_params["num_blocks"]):
            counter += i
            x = self.mb_conv_block(x, stem_params["num_filters"]*core_params["width_exp"]**counter, 
                                   core_params["kernel_size"], 
                                   core_params["strides"])
        
        # Head
        for i in range(head_params["num_layers"]):
            x  =  tf.keras.layers.Dense(head_params["num_filters"]*core_params["width_exp"]**counter, kernel_initializer = head_params["kernel_init"])(x)

        x  =  layers.GlobalAveragePooling2D()(x)#

        if self.num_classes > 2:
            x = layers.Dense(self.num_classes, activation = "softmax")(x)
        else:
            x  =  layers.Dense(1, activation = "sigmoid")(x)

        return Model(inputs, x)
    
    def compile_model(self):

        if self.num_classes > 2:
            loss = SparseCategoricalCrossentropy()
        else:
            loss = BinaryCrossentropy()

        match self.optmiser['type']:
            case "Adam":
                optimizer = Adam(learning_rate = self.optmiser['initial_lr'])

            case "Nadam":
                optimizer = Nadam(learning_rate = self.optmiser['initial_lr'])

            case "SGD":
                optimizer = SGD(learning_rate = self.optmiser['initial_lr'], momentum = self.optmiser['momentum'])

        # Compile the model with the Adam optimizer and binary crossentropy loss
        self.compile(
            optimizer = optimizer,
            loss = loss,
            metrics=['accuracy', Recall() , Precision()]
        )

    def fit_model(self, df_train: pd.DataFrame, df_test: pd.DataFrame):

        X_train = df_train.drop(columns = ['label'])
        X_test = df_test.drop(columns = ['label'])
        
        y_train = df_train['label']
        y_test = df_test['label']

        # Callbacks
        reduce_lr = ReduceLROnPlateau(
            factor = self.reduce_plateau['factor'],
            patience = self.reduce_plateau['patience'],
            verbose = 1
        )

        model_checkpoint = ModelCheckpoint(
            filepath = os.path.join("models", f'{self.config["task_name"]}_best_weights.keras'),
            monitor = "val_loss",
            save_best_only = True,
            verbose = 1
        )

        early_stopping = EarlyStopping(
            monitor = "val_loss",
            patience = self.model_params['patience'],
            min_delta = self.model_params['min_delta'],
            restore_best_weights=True
        )

        clearml_callback = ClearML.ClearMLCallback(self.clearML)

        # Fit the model
        hist = self.fit(
            X_train,
            y_train,
            epochs = self.model_params['epochs'],
            batch_size = self.model_params['batch_size'],
            validation_data = (X_test, y_test),
            callbacks = [early_stopping, reduce_lr, model_checkpoint, clearml_callback],
            verbose = 1
        )

        return hist
