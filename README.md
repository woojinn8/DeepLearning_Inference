# Introduction
C++ Inference example for Deep Learning Framework

# 1. Requirement
- CMake(v3.16.3 is used)
- OpenCV(v3.4.16 is used)
- NDK (r21e is used)


# 2.Quick Start
1. Download prebuild library which you want to use.
|Framework|Download Link|
|:---:|:---:|
|Tensorflow|[download]()|
|TFLite|[download]()|
|Pytorch|[download]()|
|MXNet|[download]()|
|MNN|[download]()|
|NCNN|[download]()|
|OpenCV|[download]()|
 
2. Build inference code using shell scripts.
  - Linux build
	- `sh build_linux.sh`
  - Android build
	- `sh build_android.sh`

# 3.Supported

## Targeet Platform
 - Linux(x64)
 - Android(aarch64)

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

## Status
|Framework|Linux|Android|Windows|
|:---:|:---:|:---:|:---:|
|Tensorflow|:white_check_mark:|-|:white_check_mark:|
|TFLite|:heavy_check_mark:|:heavy_check_mark:|-|
|Pytorch|-|-|-|
|MXNet|:white_check_mark:|-|:white_check_mark:|
|MNN|:heavy_check_mark:|:heavy_check_mark:|:white_check_mark:|
|NCNN|:heavy_check_mark:|:heavy_check_mark:|:white_check_mark:|
|OpenCV|:heavy_check_mark:|-|:heavy_check_mark:|

