#Save tflite model
import numpy as np
import tensorflow as tf
import tensorflow.lite as tflite

# Load trained model
model = tf.keras.models.load_model("mug_model.h5")

# Number of calibration steps
num_calibration_steps = 100  

# Convert model to TFLite with full integer quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset function (for images)
IMG_SIZE = (32, 32)  # Match input size of CNN
def representative_dataset_gen():
    for _ in range(num_calibration_steps):
        data = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 1).astype(np.float32)  # Simulated image
        yield [data]

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8] 

# Ensure 8-bit integer inputs/outputs
converter.inference_input_type = tf.int8  
converter.inference_output_type = tf.int8  

# Convert and save model
tflite_quant_model = converter.convert()
tflite_model_filename = "mug_model.tflite"

with open(tflite_model_filename, "wb") as fpo:
    fpo.write(tflite_quant_model)

print(f"Saved quantized TFLite model as {tflite_model_filename}")