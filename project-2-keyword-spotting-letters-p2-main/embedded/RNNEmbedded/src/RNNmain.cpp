#include <TensorFlowLite.h>
#include <Arduino.h>
#include "main_functions.h"
#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <unistd.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"

namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}

void setup() {
  delay(5000);
  Serial.begin(9600);
  while (!Serial) {}

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<14> micro_op_resolver(error_reporter);
  if (micro_op_resolver.AddAveragePool2D() != kTfLiteOk ||
      micro_op_resolver.AddConv2D() != kTfLiteOk ||
      micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk ||
      micro_op_resolver.AddFullyConnected() != kTfLiteOk ||
      micro_op_resolver.AddMaxPool2D() != kTfLiteOk ||
      micro_op_resolver.AddMul() != kTfLiteOk ||
      micro_op_resolver.AddRelu() != kTfLiteOk ||
      micro_op_resolver.AddReshape() != kTfLiteOk ||
      micro_op_resolver.AddSoftmax() != kTfLiteOk ||
      micro_op_resolver.AddAdd() != kTfLiteOk) {
    return;
  }

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) ||
      (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != kFeatureSliceCount) ||
      (model_input->dims->data[2] != kFeatureSliceSize) ||
      (model_input->dims->data[3] != 1) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    Serial.println("Input tensor parameters are incorrect. ");
    if (model_input->dims->size != 2) {
         Serial.println("Size is not 2. It is " + String(model_input->dims->size));
    }
    if (model_input->dims->data[0] != 1) {
         Serial.println("First dimension is not 1. ");
    }
    if (model_input->dims->data[1] != (kFeatureSliceCount * kFeatureSliceSize)) {
         Serial.println("Second dimension is not 1960. It is "  + String(model_input->dims->data[1]));
    }
    if (model_input->type != kTfLiteInt8) {
        Serial.println("Type is not kTfLiteInt8. ");
    }
    printf("\n");
    return;
  }
  Serial.println("Passed input tensor parameters. ");

  model_input_buffer = model_input->data.int8;
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;
  Serial.println("Made feature provider");

  static RecognizeCommands static_recognizer(error_reporter);
  recognizer = &static_recognizer;

  previous_time = 0;
  delay(5000); 
  Serial.println("Exit Setup");
}

void loop() {
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      error_reporter, previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Feature generation failed");
    return;
  }
  previous_time = current_time;

  if (how_many_new_slices == 0) {
    return;
  }

  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  uint32_t start_time = micros();

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);

  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "RecognizeCommands::ProcessLatestResults() failed");
    return;
  }

  RespondToCommand(error_reporter, current_time, found_command, score, is_new_command);
  
  uint32_t end_time = micros();
  uint32_t inference_time = end_time - start_time;
  float inference_time_ms = inference_time / 1000.0;

  // Print inference time
  //Serial.print("Inference time: ");
  //Serial.print(inference_time_ms);
  //Serial.println(" ms");  
}
