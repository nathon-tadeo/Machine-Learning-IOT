import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, MaxPooling2D, Flatten, Dense, Input, Add, Activation
from tensorflow.keras.models import Model
import numpy as np

def build_model1():
  model = Sequential([
  layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
  layers.BatchNormalization(),
  
  layers.Conv2D(64, (3,3), strides=2, padding='same', activation='relu'),
  layers.BatchNormalization(),
  
  layers.Conv2D(128, (3,3), strides=2, padding='same', activation='relu'),
  layers.BatchNormalization(),
  
  layers.Conv2D(128, (3,3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  
  layers.Conv2D(128, (3,3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  
  layers.Conv2D(128, (3,3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  
  layers.Conv2D(128, (3,3), padding='same', activation='relu'),
  layers.BatchNormalization(),
  
  layers.MaxPooling2D(pool_size=(4,4), strides=4),
  layers.Flatten(),
  
  layers.Dense(128, activation='relu'),
  layers.BatchNormalization(),
  
  layers.Dense(10, activation='softmax')
])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

def build_model2():
  model = Sequential([
    layers.Conv2D(32, (3,3), strides=2, padding='same', activation='relu', input_shape=(32,32,3)),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    layers.SeparableConv2D(128, (3,3), strides=2, padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    layers.DepthwiseConv2D((3,3), padding='same', strides=1, use_bias=False),
    layers.Conv2D(128, (1,1), strides=1, activation='relu'),
    layers.BatchNormalization(),
    
    layers.DepthwiseConv2D((3,3), padding='same', strides=1, use_bias=False),
    layers.Conv2D(128, (1,1), strides=1, activation='relu'),
    layers.BatchNormalization(),
    
    layers.DepthwiseConv2D((3,3), padding='same', strides=1, use_bias=False),
    layers.Conv2D(128, (1,1), strides=1, activation='relu'),
    layers.BatchNormalization(),
    
    layers.DepthwiseConv2D((3,3), padding='same', strides=1, use_bias=False),
    layers.Conv2D(128, (1,1), strides=1, activation='relu'),
    layers.BatchNormalization(),
    
    layers.MaxPooling2D(pool_size=(4,4), strides=4),
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dense(10, activation='softmax')
])
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  return model

def build_model3():
    inputs = layers.Input(shape=(32, 32, 3))
    
    # First convolutional block
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Second convolutional block
    residual = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Dropout(0.3)(residual)

    # Skip connection 1
    if x.shape[-1] != residual.shape[-1]:
        x = layers.Conv2D(64, (1, 1), strides=2, padding='same')(x)
    x = layers.add([x, residual])
    
    # Third convolutional block
    residual = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Dropout(0.3)(residual)
    
    # Fourth convolutional block
    residual = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(residual)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Dropout(0.3)(residual)

    # Skip connection 2
    if x.shape[-1] != residual.shape[-1]:
        x = layers.Conv2D(128, (1, 1), strides=2, padding='same')(x)
    x = layers.add([x, residual])
    
    # Fifth convolutional block
    residual = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Dropout(0.3)(residual)
    
    # Sixth convolutional block
    residual = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(residual)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Dropout(0.3)(residual)
    
    # Skip connection 3
    if x.shape[-1] != residual.shape[-1]:
        x = layers.Conv2D(128, (1, 1), padding='same')(x)
    x = layers.add([x, residual])

    # Seventh convolutional block
    residual = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Dropout(0.3)(residual)
    
    # Global pooling and dense layers
    x = layers.MaxPooling2D(pool_size=(4,4), strides=4)(residual)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model3 = Model(inputs, outputs)
    
    # Compile the model
    model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model3

def build_model50k():
  model = Sequential([
      layers.Conv2D(16, (3,3), strides=1, padding='same', activation='relu', input_shape=(32,32,3)),
      layers.BatchNormalization(),
      
      layers.SeparableConv2D(32, (3,3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      
      layers.SeparableConv2D(64, (3,3), strides=2, padding='same', activation='relu'),
      layers.BatchNormalization(),
      
      layers.SeparableConv2D(128, (3,3), strides=2, padding='same', activation='relu'),
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
if __name__ == '__main__':
# Load CIFAR-10 dataset
  (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
  # Normalize images
  train_images, test_images = train_images / 255.0, test_images / 255.0
  # Split training set into training and validation sets
  val_images, train_images = train_images[:10000], train_images[10000:]
  val_labels, train_labels = train_labels[:10000], train_labels[10000:]


  # Build model 1
  model1 = build_model1()
  model1.summary()
  model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
  test_loss, test_acc = model1.evaluate(test_images, test_labels)
  print(f'Test accuracy: {test_acc:.4f}')

  # Load and preprocess test image (Resized car image)
  img_path = 'test_image_classname.jpg'
  img = image.load_img(img_path, target_size=(32, 32))
  img_array = image.img_to_array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)

  # Run inference on test
  predictions = model1.predict(img_array)
  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  predicted_class = class_names[np.argmax(predictions)]
  print(f'Predicted class: {predicted_class}')



  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()
  model2.summary()
  model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
  test_loss, test_acc = model2.evaluate(test_images, test_labels)
  print(f'Test accuracy: {test_acc:.4f}')



  ### Repeat for model 3
  model3 = build_model3()
  model3.summary()
  model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
  test_loss, test_acc = model3.evaluate(test_images, test_labels)
  print(f'Test accuracy: {test_acc:.4f}')



  ### Repeat best sub-50k params model
  model50k = build_model50k()
  model50k.summary()
  model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels), batch_size=64)
  test_loss, test_acc = model50k.evaluate(test_images, test_labels)
  print(f'Test accuracy: {test_acc:.4f}')
  model50k.save("best_model.h5")