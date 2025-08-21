import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load trained model
model = tf.keras.models.load_model("mug_model.h5")

# Dataset directory
data_dir = "dataset"

# Image size and batch size
IMG_SIZE = (32, 32)
BATCH_SIZE = 8

# Data Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation datasets
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='training',
    color_mode='grayscale'
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    subset='validation',
    color_mode='grayscale'
)

# Evaluate model on training and validation sets
train_loss, train_accuracy = model.evaluate(train_data)
val_loss, val_accuracy = model.evaluate(val_data)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Load saved history (Check if file exists)
try:
    history = np.load("mug_history.npy", allow_pickle=True).item()
    loss = history['loss']
    val_loss = history['val_loss']

    # Plot training vs validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(loss, label="Training Loss", color='blue')
    plt.plot(val_loss, label="Validation Loss", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()
except FileNotFoundError:
    print("No training history file found. Cannot plot loss.")