import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing import image

def build_model1():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), strides=2, padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.MaxPooling2D(pool_size=(4, 4), strides=4),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model2():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), strides=2, padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(64, (3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(128, (3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.MaxPooling2D(pool_size=(4, 4), strides=4),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_model3():
    inputs = keras.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # First residual block
    residual = x
    x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    residual = layers.Conv2D(64, (1, 1), strides=2, padding="same")(residual)  # Adjust channel size
    x = layers.Add()([x, residual])
    
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)  # Dropout added
    
    # Second residual block
    residual = x
    x = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Skip connection
    x = layers.Add()([x, residual])
    
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs, x)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def build_model50k():
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), strides=1, padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(32, (3, 3), strides=1, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(64, (3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(64, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),  # Reduces parameters
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# no training or dataset construction should happen above this line
if __name__ == "__main__":
    ########################################
    ## Add code here to Load the CIFAR10 data set
# Load CIFAR-10 dataset
    (x_train, y_train), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Normalize images to [0, 1] range
    x_train, test_images = x_train / 255.0, test_images / 255.0

# Split training set into training and validation subsets
    train_images, val_images, train_labels, val_labels = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

    ########################################
    ## Build and train model 1
model1 = build_model1()
model1.summary()
    # Compile and train model 1.
history = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

# Evaluate on test data
test_loss, test_acc = model1.evaluate(test_images, test_labels)

# Print final accuracies
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

    ## Build, compile, and train model 2 (DS Convolutions)
model2 = build_model2()
model2.summary()
    # Compile and train model 2.
history = model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

# Evaluate on test data
test_loss, test_acc = model2.evaluate(test_images, test_labels)

# Print final accuracies
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

    ### Repeat for model 3 and your best sub-50k params model
model3 = build_model3()
model3.summary()
    # Compile and train model 3.
history = model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

# Evaluate on test data
test_loss, test_acc = model3.evaluate(test_images, test_labels)

# Print final accuracies
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

    ### Repeat for model 3 and your best sub-50k params model
model50k = build_model50k()
model50k.summary()
    # Compile and train model 3.
history = model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

# Evaluate on test data
test_loss, test_acc = model50k.evaluate(test_images, test_labels)

# Print final accuracies
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")
print(f"Final Test Accuracy: {test_acc:.4f}")

########################################
    # Load the image
img_path = "test_image_dog.png"  # Replace with actual image filename
img = image.load_img(img_path, target_size=(32, 32))

    # Convert image to array and normalize
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the trained model
predictions = model1.predict(img_array)
predicted_class = np.argmax(predictions)

    # CIFAR-10 class labels
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck']

predicted_label = class_labels[predicted_class]
expected_class = "dog" 
print(f"Predicted class: {predicted_label}")
print(f"Correct label: {expected_class}")
print(f"Prediction correct? {predicted_label == expected_class}")