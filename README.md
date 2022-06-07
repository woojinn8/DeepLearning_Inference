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

# 3.Supported Framework
Currently, it's supports Linux and Android.

- [ ] Tensorflow
- [ ] MXNet
- [x] TFLite
- [x] NCNN
- [x] MNN
- [ ] onnxruntime
- [ ] OpenCV




