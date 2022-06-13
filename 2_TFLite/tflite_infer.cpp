#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

#include "tensorflow/lite/delegates/gpu/delegate.h"

using namespace tflite;

class tflite_engine
{
  public:
    tflite_engine() {};
    ~tflite_engine() {
      model.release();
      interpreter.release();
    };
    bool Initialize(std::string model_file,
                        int use_gpu,
                        int use_fp16,
                        int num_thread);
    bool predict(cv::Mat img);

  private:
    // Load model
    std::unique_ptr<tflite::FlatBufferModel> model;

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    TfLiteDelegate *delegate;

    int height = 112;
    int width = 112;

};

bool tflite_engine::Initialize(std::string model_file, 
                                int use_gpu,
                                int use_fp16,
                                int num_thread)
{
  this->model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  tflite::InterpreterBuilder(*this->model, this->resolver)(&this->interpreter);

  int result_delegate;
  if (use_gpu)
  {
    auto options = TfLiteGpuDelegateOptionsV2Default();
    options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    if (use_fp16)
    {
      options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
      options.is_precision_loss_allowed = 1;
    }
    else
    {
      options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
    }
    this->delegate = TfLiteGpuDelegateV2Create(&options);
    result_delegate = this->interpreter->ModifyGraphWithDelegate(this->delegate);
  }

  int result_set_thread = this->interpreter->SetNumThreads(num_thread);
  int result_allocate = this->interpreter->AllocateTensors();

  if (result_delegate != 0 || result_set_thread != 0 || result_allocate != 0)
  {
    std::cout << "settting result : " << result_delegate << ", " << result_set_thread << ", " << result_allocate << std::endl;
    return false;
  }

  return true;
}

bool tflite_engine::predict(cv::Mat img)
{
  try
  {
    // input is correct!! - 2021-11-08 17:28
    // Set input
    float *input = interpreter->typed_input_tensor<float>(0);
    for (int y = 0; y < height; y++)
    {
      for (int x = 0; x < width; x++)
      {
        *(input + y * width + x) = ((float)img.data[3 * y * width + 3 * x] * 0.0078125 - 127.5 * 0.0078125);
        *(input + y * width + x + 1 * height * width) = ((float)img.data[3 * y * width + 3 * x + 1] * 0.0078125 - 127.5 * 0.0078125);
        *(input + y * width + x + 2 * height * width) = ((float)img.data[3 * y * width + 3 * x + 2] * 0.0078125 - 127.5 * 0.0078125);
      }
    }

    // iter inference
    float total_runtime = 0;
    float max_runtime = 0;
    int total_iter = 30;
    for (int i = 0; i < total_iter; i++)
    {
      TfLiteTensor *output_tensor = nullptr;

      std::chrono::system_clock::time_point begin_time = std::chrono::system_clock::now();
      this->interpreter->Invoke();
      std::chrono::duration<double> sec = std::chrono::system_clock::now() - begin_time;

      // check outut
      output_tensor = this->interpreter->tensor(this->interpreter->outputs()[0]);

      if (i == 0)
        continue; // warmp-up

      total_runtime += sec.count() * 1000;

      if (max_runtime < sec.count() * 1000)
      {
        max_runtime = sec.count() * 1000;
      }
    }

    float average_runtime = total_runtime / total_iter;
    std::cout << average_runtime << " / " << max_runtime << std::endl;
  } 
  catch (int expn)
  {
    std::cout << "Fail to Inference" << std::endl;
    return false;
  }

  return true;

}

int main(int argc, char **argv)
{

  // set input size
  int width = 112;
  int height = 112;

  std::cout << "argv 1 : model_file" << std::endl;
  std::cout << "argv 2 : input_image" << std::endl;

  std::string model_file = argv[1];
  std::string input_img_file = argv[2];

  cv::Mat img = cv::imread(input_img_file);
  cv::resize(img, img, cv::Size(width, height));
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  printf("compete load image\n");

  tflite_engine *tflite_ptr = new tflite_engine;
  bool result_init = tflite_ptr->Initialize(model_file, 1, 1, 1);
  if(!result_init)
  {
    printf("Fail to Initialize\n");
    return -1;
  }

  bool result_predict = tflite_ptr->predict(img);
  if(!result_predict)
  {
    printf("Fail to predict\n");
    return -1;
  }
  
  return 0;
}