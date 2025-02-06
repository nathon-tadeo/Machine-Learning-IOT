import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_iris

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
##


def build_model1():
    model = None  # Add code to define model 1.
    return model


def build_model2():
    model = None  # Add code to define model 1.
    return model


def build_model3():
    model = None  # Add code to define model 1.
    ## This one should use the functional API so you can create the residual connections
    return model


def build_model50k():
    model = None  # Add code to define model 1.
    return model


# no training or dataset construction should happen above this line
if __name__ == "__main__":
    ########################################
    ## Add code here to Load the CIFAR10 data set
   # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images

    # Build and compile the model
    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    


    ########################################
    ## Build and train model 1
    model1 = build_model1()
    # compile and train model 1.

    ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()

    ### Repeat for model 3 and your best sub-50k params model
