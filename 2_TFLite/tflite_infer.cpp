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

  std::cout << "01. complete load model" << std::endl;

  std::string output_file_name = model_file.substr(0, model_file.size() - 7) + ".csv";
  std::cout << "output file name : " << output_file_name << std::endl;
  std::ofstream result_output;

  result_output.open(output_file_name);
  result_output << "Platform,FP16,#Thread,Affinity,Avg_time,Max_time\n";
  result_output.close();

  // iter use gpu
  for (int iter_gpu = 1; iter_gpu < 2; iter_gpu++)
  {
    std::string platform;
    if (iter_gpu == 0)
      platform = "CPU";
    else
      platform = "GPU";

    // iter num thread
    for (int iter_thread = 4; iter_thread <= 6; iter_thread++)
    {

      if (iter_gpu == 1 && iter_thread > 4)
        break;

      // iter fp16
      for (int iter_fp16 = 0; iter_fp16 < 2; iter_fp16++)
      {

        std::string fp16;
        if (iter_fp16 == 0)
          fp16 = "OFF";
        else
          fp16 = "ON ";

        try
        {

          // Load model
          std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());

          // Build the interpreter
          tflite::ops::builtin::BuiltinOpResolver resolver;
          std::unique_ptr<tflite::Interpreter> interpreter;
          tflite::InterpreterBuilder(*model, resolver)(&interpreter);

          // set delegate option
          TfLiteDelegate *delegate;
          int result_delegate;
          if (iter_gpu)
          {
            auto options = TfLiteGpuDelegateOptionsV2Default();
            options.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
            if (iter_fp16)
            {
              options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
              options.is_precision_loss_allowed = 1;
            }
            else
            {
              options.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
            }
            delegate = TfLiteGpuDelegateV2Create(&options);
            result_delegate = interpreter->ModifyGraphWithDelegate(delegate);
          }

          int result_set_thread = interpreter->SetNumThreads(iter_thread);
          int result_allocate = interpreter->AllocateTensors();

          if (result_delegate != 0 || result_set_thread != 0 || result_allocate != 0)
          {
            std::cout << "settting result : " << result_delegate << ", " << result_set_thread << ", " << result_allocate << std::endl;
            continue;
          }

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
          for(int i=0; i < total_iter; i++)
          {
            TfLiteTensor *output_tensor = nullptr;

            std::chrono::system_clock::time_point begin_time = std::chrono::system_clock::now();
            interpreter->Invoke();
            std::chrono::duration<double> sec = std::chrono::system_clock::now() - begin_time;

            //check outut
            output_tensor = interpreter->tensor(interpreter->outputs()[0]);

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
          result_output.open(output_file_name, std::ios::app);
          result_output << platform << "," << fp16 << "," << iter_thread << ",-," << average_runtime << "," << max_runtime << "\n";
          result_output.close();

          model.release();
          interpreter.release();
        } //end iter fp16
        catch (int expn)
        {
          std::cout << "Fail in " + model_file + " - " << platform << " / " << fp16 << " / " << iter_thread << std::endl;
          result_output << "\n";
        }
      }

    } // end iter num thread

  } // end iter use gpu

  return 0;
}