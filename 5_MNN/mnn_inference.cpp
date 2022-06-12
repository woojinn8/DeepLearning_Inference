#include <iostream>

#include "opencv2/opencv.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

#include <chrono>
#include <iostream>
#include <fstream>
#include <dlfcn.h>


class mnn_engine{

public:
    mnn_engine() {};
	~mnn_engine() {};

	// MNN
    bool Initialize_mnn(std::string model_file, 
                        int use_gpu,
                        int num_thread_, 
                        float threshold
                        );
	int predict(cv::Mat image);
private:
   float threshold;

	std::shared_ptr<MNN::Interpreter> interpreter;
	MNN::Session *session = nullptr;
	MNN::Tensor *inputTensor = nullptr;
	MNN::Tensor *outputTensor = nullptr;

	MNN::ScheduleConfig config_;
	MNN::BackendConfig backendConfig_;

    float pixel_mean_mask[3] = {127.5, 127.5, 127.5};
    float pixel_std_mask[3] = {1.0/128.0, 1.0/128.0, 1.0/128.0};
    int input_size = 128;
};


bool mnn_engine::Initialize_mnn(std::string model_file, 
                                int use_gpu,
                                int num_thread_, 
                                float threshold)
{
	this->interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
	MNN::ScheduleConfig config;
    if(use_gpu)
        config.type = MNN_FORWARD_VULKAN;
    else
        config.type = MNN_FORWARD_CPU;

	config.numThread = num_thread_;
	
    MNN::BackendConfig backendConfig;
    backendConfig.memory = MNN::BackendConfig::MemoryMode::Memory_High;
	backendConfig.precision = MNN::BackendConfig::PrecisionMode::Precision_High;
    backendConfig.power = MNN::BackendConfig::PowerMode::Power_High;

	config.backendConfig = &backendConfig;

	this->session = interpreter->createSession(config);
	this->inputTensor = interpreter->getSessionInput(session, nullptr);
	this->threshold = threshold;

    return true;
}


int mnn_engine::predict(cv::Mat image)
{

   if (image.empty())
	{
		std::cout << "image is empty ,please check!" << std::endl;
		return -1;
	}
	//cv::Mat image;
	//cv::resize(raw_image, image, cv::Size(target_image_width, target_image_height));
	this->interpreter->resizeTensor(inputTensor, {1, 3, input_size, input_size});
	this->interpreter->resizeSession(this->session);

	std::shared_ptr<MNN::CV::ImageProcess> pretreat(
		MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, pixel_mean_mask, 3, pixel_std_mask, 3));
	pretreat->convert(image.data, input_size, input_size, image.step[0], this->inputTensor);

	this->interpreter->runSession(this->session);

	this->outputTensor = this->interpreter->getSessionOutput(this->session, "fc7");
	MNN::Tensor output_host(this->outputTensor, this->outputTensor->getDimensionType());
	this->outputTensor->copyToHostTensor(&output_host);
	float *output_data = output_host.host<float>();

	cv::Mat res_ori = cv::Mat(1, 2, CV_32F, output_data);

	float exp_float[2];
	float sum_exp = 0.0;
	for (int i = 0; i < 2; i++)
	{
		exp_float[i] = exp(res_ori.at<float>(i));
		sum_exp += exp_float[i];
	}

	cv::Mat res(1, 2, CV_32FC1);

	for (int i = 0; i < 2; i++)
		res.at<float>(i) = (exp_float[i] / sum_exp);

	std::cout << res.at<float>(0, 0) << ", " << res.at<float>(0, 1) << std::endl;

	// 04 Check result
	int result = -1;
	if (res.at<float>(0, 0) < res.at<float>(0, 1)) // is-half?
		result = 1;
	else
		result = 0;
	
	return result;


}

int main(int argc, char** argv)
{

    std::cout << "1. model_file" << std::endl;
    std::cout << "2. use_gpu" << std::endl;
    std::cout << "3. num_thread" << std::endl;


    auto handle = dlopen("libMNN_Vulkan.so", RTLD_NOW);

    mnn_engine engine;

    std::string model_file = argv[1];
    int use_gpu = std::atoi(argv[2]);
    int num_thread = std::atoi(argv[3]);


    bool init_result = engine.Initialize_mnn(model_file, use_gpu, num_thread, 0.5);
    int result = engine.predict(cv::Mat::zeros(cv::Size(128, 128), CV_8UC3));

}