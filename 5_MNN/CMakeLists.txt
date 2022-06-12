cmake_minimum_required(VERSION 3.10)
project(mnn_inference C CXX)

OPTION (USE_ANDROID "Build for Android" OFF)

# Basic Path Setting
set(ATTRIBUTE PRIVATE)
set(THIRD_PARTY ${CMAKE_CURRENT_LIST_DIR}/../0_3rdparty)

if(USE_ANDROID)
set(BUILD_PATH ${CMAKE_CURRENT_LIST_DIR}/build_android)
else()
set(BUILD_PATH ${CMAKE_CURRENT_LIST_DIR}/build_linux)
endif()

if(USE_ANDROID)
set(CMAKE_EXE_LINKER_FLAGS    "-Wl,--as-needed, -fuse-ld=gold ${CMAKE_EXE_LINKER_FLAGS}")
else()
set(CMAKE_EXE_LINKER_FLAGS    "-Wl,--as-needed ${CMAKE_EXE_LINKER_FLAGS}")
endif()


# MNN Path Setting
set(MNN_INC ${THIRD_PARTY}/mnn/include)
if(USE_ANDROID)
set(MNN_LIB ${THIRD_PARTY}/mnn/libs/android_aarch64/libMNN.so)
set(MNN_VULKAN_LIB ${THIRD_PARTY}/mnn/libs/android_aarch64/libMNN_Vulkan.so)
else()
set(MNN_LIB ${THIRD_PARTY}/mnn/libs/linux_x64/libMNN.so)
set(MNN_VULKAN_LIB ${THIRD_PARTY}/mnn/libs/linux_x64/libMNN_Vulkan.so)
endif()

set(VULKAN_INC ${THIRD_PARTY}/mnn/vulkan)

# OpenCV Path Setting
if(USE_ANDROID)
set(OPENCV_INC ${THIRD_PARTY}/opencv/opencv_android_aarch64/sdk/native/jni/include)
set(OpenCV_LIBS ${THIRD_PARTY}/opencv/opencv_android_aarch64/sdk/native/libs/arm64-v8a)
else()
set(OPENCV_INC ${THIRD_PARTY}/opencv/opencv_linux_x64/include)
set(OpenCV_LIBS ${THIRD_PARTY}/opencv/opencv_linux_x64/libs)
endif()

add_executable(mnn_inference 
${CMAKE_CURRENT_LIST_DIR}/mnn_inference.cpp
)

# Link MNN
target_include_directories(mnn_inference ${ATTRIBUTE} ${MNN_INC})
target_link_libraries(mnn_inference ${MNN_LIB} ${MNN_VULKAN_LIB})

# Link OpenCV
target_include_directories(mnn_inference ${ATTRIBUTE} ${OPENCV_INC})
target_link_libraries(mnn_inference
  ${OpenCV_LIBS}/libopencv_highgui.so
  ${OpenCV_LIBS}/libopencv_video.so
  ${OpenCV_LIBS}/libopencv_photo.so
  ${OpenCV_LIBS}/libopencv_features2d.so
  ${OpenCV_LIBS}/libopencv_core.so
  ${OpenCV_LIBS}/libopencv_imgproc.so
  -ldl
  )
  

if(USE_ANDROID)
target_include_directories(mnn_inference ${ATTRIBUTE} ${VULKAN_INC})
find_library(ANDROID_LOG_LIB log)
target_link_libraries(mnn_inference log)
endif()
