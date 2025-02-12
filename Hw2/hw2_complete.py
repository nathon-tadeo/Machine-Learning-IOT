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
from tensorflow.keras.models import load_model

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
    return model

def build_model3():
    inputs = keras.Input(shape=(32, 32, 3))

    # Initial convolution layer (no residual connection)
    x = layers.Conv2D(32, (3, 3), strides=2, padding="same", activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    # First residual block (Conv2D → BatchNorm → Dropout) x2
    residual = x
    x = layers.Conv2D(64, (3, 3), strides=2, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    x = layers.Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    # Skip connection (1x1 conv to match channels)
    residual = layers.Conv2D(64, (1, 1), strides=2, padding="same")(residual)
    x = layers.Add()([x, residual])

    # Second residual block (Conv2D → BatchNorm → Dropout) x2
    residual = x
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    x = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    # Skip connection (1x1 conv to match channels)
    residual = layers.Conv2D(128, (1, 1), strides=2, padding="same")(residual)
    x = layers.Add()([x, residual])

    # Third residual block (Conv2D → BatchNorm → Dropout) x2
    residual = x
    x = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    x = layers.Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout

    # Skip connection (no need for 1x1 conv since channel counts match)
    x = layers.Add()([x, residual])

    # Pooling & Fully Connected Layers
    x = layers.MaxPooling2D(pool_size=(4, 4), strides=4)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)  # Added dropout
    x = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, x)
    return model

def build_model50k():
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), strides=1, padding="same", activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(32, (3, 3), strides=1, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(64, (3, 3), strides=2, padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.SeparableConv2D(128, (3, 3), padding="same", activation='relu'),
        layers.BatchNormalization(),
        
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# no training or dataset construction should happen above this line
if __name__ == "__main__":
    ########################################
    ## Load the CIFAR-10 dataset and split
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize images
    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()

    # Split training data into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    ########################################
    ## Build and train model 1
    model1 = build_model1()
    model1.summary()
    # Compile and train Model 1
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Ensure train_images is accessible
    print(f"train_images shape: {train_images.shape}")  # Debugging step
    history = model1.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))

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
    # Compile and train Model 2
    model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Ensure train_images is accessible
    print(f"train_images shape: {train_images.shape}")  # Debugging step
    history = model2.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))

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
    # Compile and train Model 3
    model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Ensure train_images is accessible
    print(f"train_images shape: {train_images.shape}")  # Debugging step
    history = model3.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))

    # Evaluate on test data
    test_loss, test_acc = model3.evaluate(test_images, test_labels)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    ## Build, compile, and train model sub-50k
    model50k = build_model50k()
    model50k.summary()
    model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
    history = model50k.fit(train_images, train_labels, epochs=1, validation_data=(val_images, val_labels))
    test_loss, test_acc = model50k.evaluate(test_images, test_labels)
    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    
    print(f"Final Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")
    print(f"Final Test Accuracy: {test_acc:.4f}")
    model50k.save("best_model.h5")
########################################
'''
# Load trained model before prediction
model1 = load_model("model1.h5")

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
'''