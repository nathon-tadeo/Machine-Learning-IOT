import tensorflow as tf
import sys
import subprocess

print("Python version:", sys.version)
print("TensorFlow version:", tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

tf.config.list_physical_devices('GPU')

tf.test.is_gpu_available()