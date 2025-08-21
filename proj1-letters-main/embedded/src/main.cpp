#include <TensorFlowLite.h>
#include <Arduino.h>
#include <Arduino_OV767X_TinyMLx.h>

#include "mug_model.h"
#include "main_functions.h"
#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

static tflite::ErrorReporter* error_reporter = nullptr;
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 180 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  while (!Serial);

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  //Initialize the camera
  if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

    //Load Model
    model = tflite::GetModel(mug_detect_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      TF_LITE_REPORT_ERROR(error_reporter,
                           "Model provided is schema version %d not equal "
                           "to supported version %d.",
                           model->version(), TFLITE_SCHEMA_VERSION);
      return;
    }
    Serial.println("Model loaded.");
  
    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddDepthwiseConv2D();
    micro_op_resolver.AddLogistic();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddMean();
    Serial.println("Ops resolver created.");
  
    //Create the TFLITE Interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;
    Serial.println("Interpreter created.");
  
    //Allocate Memory for Tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
      return;
    }
    Serial.println("Memory allocated.");
  
    //Get a Pointer to the Input Tensor
    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("Initialization done.");
}

void loop() {
// Capture an Image (GetImage)
 if (kTfLiteOk != GetImage(error_reporter, kNumCols, kNumRows, kNumChannels,
                            input->data.int8)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Image capture failed.");
  }

  // Run inference
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }

  // Process output
  TfLiteTensor* output = interpreter->output(0);

  int8_t mug_score = output->data.uint8[kMugIndex];
  int8_t no_mug_score = output->data.uint8[kNotAMugIndex];
  RespondToDetection(error_reporter, mug_score, no_mug_score);
}