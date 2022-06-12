# Introduction
C++ Inference example for Deep Learning Framework

# 1. Requirement
- CMake
- OpenCV
- NDK (for android)


# 2.Quick Start
1. Download prebuild library which you want to use.
2. In order to link the prebuilt libs, you need to export DeepLearning_Inference/prebuilt to LD_LIBRARY_PATH
3. Build inference code using shell scripts.
  - Linux build
	- `sh build_linux.sh`
  - Android build
	- `sh build_android.sh`

# 3.Supported

## Targeet Platform

 - Linux(x64)
 - Android(aarch64)
 - Windows(x64)

## Target Framework
 - Tensorflow
 	- You may need a version newer than 2.6
 - TFLite
	- CPU and GPU are supported
	- Edge TPU, XNNPACK and NNAPI will updated soon
 - Pytorch
 - MXNet
 - MNN
 - NCNN
 - Opencv
 	- You may need a version newer than 3.1 to use DNN module

|Framework|Linux|Android|Windows|
|:---:|:---:|:---:|:---:|
|Tensorflow|:white_check_mark:|-|:heavy_check_mark:|
|TFLite|:heavy_check_mark:|:heavy_check_mark:|-|
|Pytorch|-|-|-|
|MXNet|:white_check_mark:|-|:heavy_check_mark:|
|MNN|:heavy_check_mark:|:heavy_check_mark:|:white_check_mark:|
|NCNN|:heavy_check_mark:|:heavy_check_mark:|:heavy_check_mark:|
|OpenCV|:heavy_check_mark:|-|:heavy_check_mark:|

