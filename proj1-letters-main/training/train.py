import os
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Dataset directory
data_dir = "dataset"

# Image size & batch size
IMG_SIZE = (32, 32)
BATCH_SIZE = 8

# Image Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    validation_split=0.2,  # 80% training, 20% validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Load dataset
train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    subset="training",
    color_mode='grayscale'
)

val_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
    subset="validation",
    color_mode='grayscale'
)

# Print dataset details
print("Classes found:", os.listdir(data_dir))
for category in os.listdir(data_dir):
    path = os.path.join(data_dir, category)
    print(f"{category} contains {len(os.listdir(path))} images.")

# Optimized CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 1)),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.Dropout(0.2),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),  # Replaces GlobalAveragePooling2D() for small model

    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),  # Reduce overfitting
    layers.Dense(1, activation="sigmoid"),
])
model.summary()
# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(
    train_data,
    steps_per_epoch=len(train_data),
    epochs=10,  
    validation_data=val_data,
    validation_steps=len(val_data),
)

# Save model
model.save("mug_model.h5")
print("Model saved as 'mug_model.h5'.")

# Save history as a NumPy file
np.save('mug_history.npy', history.history)