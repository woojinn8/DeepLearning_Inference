#include "gpu.h"
#include "net.h"
#include "mat.h"
#include "cpu.h"
#include "datareader.h"

#include "opencv2/opencv.hpp"

#include <chrono>
#include <iostream>
#include <fstream>

#include <unistd.h>

#define VULKAN_BUILD

class ncnn_engine
{
public:
    ncnn::Net net;

    std::string model_bin;
    std::string model_param;
    int num_thread;
    int use_gpu;
    int use_fp16;
    int power_save;

    bool Initialize(std::string bin,
                    std::string param,
                    int thread,
                    int gpu,
                    int fp16,
                    int ps)
    {
        model_bin = bin;
        model_param = param,
        num_thread = thread;
        use_gpu = gpu;
        use_fp16 = fp16;
        power_save = ps;


        /*
        std::cout << "bin : " << model_bin << std::endl;
        std::cout << "param : " << model_param << std::endl;
        std::cout << "num_threads : " << num_thread << std::endl;
        std::cout << "use_gpu : " << use_gpu << std::endl;
        std::cout << "use_fp16 : " << use_fp16 << std::endl;
        std::cout << "powersave(0=all,1=little,2=big) : " << power_save << std::endl;
        */

#ifdef VULKAN_BUILD
        if (use_gpu)
        {
            if (ncnn::get_gpu_count() > 0)
            {
                net.opt.use_vulkan_compute = 1;
            }
        }
#endif
        if (use_fp16)
        {
            net.opt.use_fp16_packed = true;
            net.opt.use_fp16_storage = true;
            net.opt.use_fp16_arithmetic = true;
        }
        else
        {
            net.opt.use_fp16_packed = false;
            net.opt.use_fp16_storage = false;
            net.opt.use_fp16_arithmetic = false;
        }

        net.load_param_enc(model_param.c_str(), 6, "s1face");
        net.load_model_enc(model_bin.c_str(), 6, "s1face");


        return true;
    }

    bool predict(cv::Mat &bgr, std::string input, std::string output)
    {
        auto st = std::chrono::steady_clock::now();
        ncnn::Extractor ex = net.create_extractor();
#ifdef VULKAN_BUILD
        if (use_gpu)
        {
            ex.set_vulkan_compute(true);
        }
#endif

        ex.set_num_threads(num_thread);

        ncnn::CpuSet ps_ = ncnn::get_cpu_thread_affinity_mask(power_save);
        ncnn::set_cpu_thread_affinity(ps_);

        ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, 112, 112);
        
        const float mean_vals[3] = {127.5, 127.5, 127.5};
        const float norm_vals[3] = {1.0 / 128.0, 1.0 / 128.0, 1.0 / 128.0};
        in.substract_mean_normalize(mean_vals, norm_vals);

        auto et = std::chrono::steady_clock::now();
		std::chrono::duration<float> dt = et - st;


        st = std::chrono::steady_clock::now();
        ex.input(input.c_str(), in);

        ncnn::Mat res_ori;
        ex.extract(output.c_str(), res_ori);
        
        et = std::chrono::steady_clock::now();
		dt = et - st;
		

        return true;
    }
};

int main(int argc, char **argv)
{
    std::string model_file = argv[1];
    std::string bin = model_file + ".bin";
    std::string param = model_file + ".param";
    std::string img_path = argv[2];

    std::string input_tensor_name = argv[3];
    std::string output_tensor_name = argv[4];

    int no_gpu_flag = 0;
    if(argc > 5)
        no_gpu_flag = std::atoi(argv[5]);

    cv::Mat img = cv::imread(img_path);
    cv::resize(img, img, cv::Size(112, 112));

    std::string output_file_name = model_file + ".csv";
    std::cout << "output file name : " << output_file_name << std::endl;
    std::ofstream result_output;
    result_output.open(output_file_name);
    result_output << "Platform,FP16,#Thread,Affinity,Avg_time,Max_time\n";
    result_output.close();

    

    for (int iter_gpu = 0; iter_gpu < 2; iter_gpu++)
    {
        if(no_gpu_flag && iter_gpu == 1)
            continue;


        std::string platform;
        if (iter_gpu == 0)
            platform = "CPU";
        else
            platform = "GPU";

        for (int iter_ps = 0; iter_ps < 3; iter_ps++)
        {   
            if(iter_gpu == 1 && iter_ps > 0)
                continue;

            std::string affinity;
            if(iter_gpu == 0)
            {
                if (iter_ps == 0)
                    affinity = "All";
                else if (iter_ps == 1)
                    affinity = "Little";
                else
                    affinity = "Big";
            }
            else
            {   
                affinity = "-";
            }

            for (int iter_thread = 1; iter_thread <= 8; iter_thread++)
            {
                if(iter_gpu == 1 && iter_thread > 4)
                    continue;

                for (int iter_fp16 = 0; iter_fp16 < 2; iter_fp16++)
                {
                    std::string fp16;
                    if (iter_fp16 == 0)
                        fp16 = "OFF";
                    else
                        fp16 = "ON ";

                    try
                    {
                        ncnn_engine *ncnn_ptr = new ncnn_engine;
                        bool result = ncnn_ptr->Initialize(bin, param, iter_thread, iter_gpu, iter_fp16, iter_ps);

                        // warm-up
                        cv::Mat tmp_img = cv::Mat::ones(cv::Size(112, 112), CV_8UC3);
                        cv::randu(tmp_img, cv::Scalar::all(0), cv::Scalar::all(255));
                        for(int i=0; i<3; i++)
                            ncnn_ptr->predict(tmp_img, input_tensor_name, output_tensor_name);

                        // iter inference
                        float total_runtime = 0;
                        float max_runtime = 0;
                        int total_iter = 30;

                        std::cout << platform << " / " << fp16 << " / " << iter_thread << " / " << affinity << " : ";

                        for (int i = 0; i < total_iter + 1; i++)
                        {
                            std::chrono::system_clock::time_point begin_time = std::chrono::system_clock::now();
                            ncnn_ptr->predict(img, input_tensor_name, output_tensor_name);
                            std::chrono::duration<double> sec = std::chrono::system_clock::now() - begin_time;

                            //printf("Elapsed: %f\n\n", sec.count() * 1000);

                            if (i == 0)
                                continue; // warmp-up

                            total_runtime += sec.count() * 1000;

                            if (max_runtime < sec.count() * 1000)
                                max_runtime = sec.count() * 1000;
                        }

                        float average_runtime = total_runtime / total_iter;

                        std::cout << average_runtime << " / " << max_runtime << std::endl;
                        
                        result_output.open(output_file_name, std::ios::app);
                        result_output << platform << "," << fp16 << "," << iter_thread << "," << affinity << "," << average_runtime << "," << max_runtime << "\n";
                        result_output.close();

                        delete ncnn_ptr;

                    }
                    catch (int expn)
                    {
                        std::cout << "Fail in " + model_file + " - " << platform << " / " << fp16 << " / " << iter_thread << " / " << affinity << "\n";
                    }

                } //end iter_fp16
            }     //end iter_thread
        }         //end iter_ps
    }             //end iter_gpu

    return 0;
}
