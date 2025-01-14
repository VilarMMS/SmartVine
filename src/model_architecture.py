import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from src.Utils.clearML import ClearMLConfig as ClearML
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

        # Passing the number of classes
        self.num_classes = num_classes

        # Unpacking optmiser config
        self.optmiser = config["optimiser"]

        # Unpacking callback configs
        self.reduce_plateau = config["reduce_plateau"]

        # Input shape
        self.input_shape  =  (config['image_size']["width"], config['image_size']["height"], config['img_channels'])
        
        # Unpacking model parameters
        self.model_params = config["model_params"]
        
        # Passing ClearML object
        self.clearML  =  clearML

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

    def mb_conv_block(self, x, filters: int, kernel_size: int, strides: int):

        expansion_factor = self.model_params["expansion_factor"]
        input_channels  =  x.shape[-1]
        expanded_channels  =  input_channels*expansion_factor

        # Expansion phase
        if expansion_factor !=  1:
            x  =  self.conv_block(x, expanded_channels, kernel_size = 1)

        # Depthwise convolution
        x  =  layers.DepthwiseConv2D(kernel_size, strides = strides, padding = "same", use_bias = False)(x)
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

        # Stem
        x  =  self.conv_block(inputs, num_filters, kernel_size = 3, strides = 2)

        # Blocks
        x  =  self.mb_conv_block(x, num_filters, kernel_size = 3, strides = 1)
        x  =  self.mb_conv_block(x, num_filters * 2, kernel_size = 3, strides = 2)
        x  =  self.mb_conv_block(x, num_filters * 4, kernel_size = 3, strides = 2)

        # Head
        x  =  self.conv_block(x, num_filters * 8, kernel_size = 1)
        x  =  layers.GlobalAveragePooling2D()(x)#

        if self.num_classes > 2:
            x  =  layers.Dense(1, activation = "softmax")(x)
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
            metrics=['accuracy', 'Recall', 'Precision']
        )

    def fit_model(self, df_train: pd.DataFrame, df_test: pd.DataFrame):

        X_train = df_train.drop(columns = ['label'])
        X_test = df_test.drop(columns = ['label'])
        
        y_train = df_train['label']
        y_test = df_train['label']

        # ReduceLROnPlateau callback
        reduce_lr = ReduceLROnPlateau(
            factor = self.reduce_plateau['factor'],
            patience = self.reduce_plateau['patience'],
            verbose = 1
        )
        
        # ModelCheckpoint callback
        model_checkpoint = ModelCheckpoint(
            filepath = os.path.join("models", f'{self.config["task_name"]}_best_weights.keras'),
            monitor = "val_loss",
            save_best_only = True,
            verbose = 1
        )

        # EarlyStopping configuration
        early_stopping = EarlyStopping(
            monitor = "val_loss",
            patience = self.model_params['patience'],
            min_delta = self.model_params['min_delta'],
            restore_best_weights = True
        )

        # Use the ClearML callback for logging during training (use the already instantiated `self.clearml`)
        clearml_callback = ClearML.ClearMLCallback(self.clearml)

        # Fit the model
        hist = self.fit(
            X_train, 
            y_train,
            epochs = self.model_params['epochs'],
            batch_size = self.model_params['batch_size'],
            validation_data = (X_test, y_test),
            callbacks = [early_stopping, reduce_lr, model_checkpoint, clearml_callback],
            verbose = 10
        )

        return hist



    