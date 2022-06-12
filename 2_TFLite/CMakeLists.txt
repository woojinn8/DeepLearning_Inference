cmake_minimum_required(VERSION 3.10)
project(TFLite_inference C CXX)

OPTION (USE_ANDROID "Build for Android" OFF)

# Basic Path Setting
set(ATTRIBUTE PRIVATE)
set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/../../3rdparty)

if(USE_ANDROID)
set(BUILD_PATH ${CMAKE_CURRENT_LIST_DIR}/build_android)
else()
set(BUILD_PATH ${CMAKE_CURRENT_LIST_DIR}/build_linux)
endif()


# TFLITE Path Setting
set(TFLITE_INC
${THIRD_PARTY}/tflite
${THIRD_PARTY}/tflite/tensorflow/lite/tools/make/downloads/flatbuffers/include
${THIRD_PARTY}/tflite/tensorflow/lite/tools/make/downloads/absl
)

if(USE_ANDROID)
set(TFLITE_LIB ${THIRD_PARTY}/tflite/lib_android/libtensorflowlite.so)
set(TFLITE_GPU_LIB ${THIRD_PARTY}/tflite/lib_android/libtensorflowlite_gpu_delegate.so)
else()
set(TFLITE_LIB ${THIRD_PARTY}/tflite/lib_linux/libtensorflowlite.so)
set(TFLITE_GPU_LIB ${THIRD_PARTY}/tflite/lib_linux/libtensorflowlite_gpu_delegate.so)
endif()


# OpenCV Path Setting
if(USE_ANDROID)
set(OPENCV_INC ${THIRD_PARTY}/opencv/opencv_android/sdk/native/jni/include)
set(OpenCV_LIBS ${THIRD_PARTY}/opencv/opencv_android/sdk/native/libs/arm64-v8a)
else()
set(OPENCV_INC ${THIRD_PARTY}/opencv/opencv_linux/include)
set(OpenCV_LIBS ${THIRD_PARTY}/opencv/opencv_linux/lib)
endif()

add_executable(TFLite_inference 
${CMAKE_CURRENT_LIST_DIR}/tflite_infer.cpp
)


# Link TFlite
target_include_directories(TFLite_inference PRIVATE ${TFLITE_INC})
target_link_libraries(TFLite_inference ${TFLITE_LIB})
target_link_libraries(TFLite_inference ${TFLITE_GPU_LIB})


# Link OpenCV
target_include_directories(TFLite_inference ${ATTRIBUTE} ${OPENCV_INC})
target_link_libraries(TFLite_inference
  ${OpenCV_LIBS}/libopencv_highgui.so
  ${OpenCV_LIBS}/libopencv_video.so
  ${OpenCV_LIBS}/libopencv_photo.so
  ${OpenCV_LIBS}/libopencv_features2d.so
  ${OpenCV_LIBS}/libopencv_core.so
  ${OpenCV_LIBS}/libopencv_imgproc.so
  )
  
 
# Link Others
find_package(OpenCL)
if(OpenCL_Found)
target_include_directories(TFLite_inference ${ATTRIBUTE} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(TFLite_inference ${OpenCL_LIBRARIES})
endif()

target_link_libraries(TFLite_inference
EGL 
GLESv2
)

if(USE_ANDROID)
target_link_libraries(TFLite_inference GLESv3)
find_library(ANDROID_LOG_LIB log)
target_link_libraries(TFLite_inference log)
endif()
