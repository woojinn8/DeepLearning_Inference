cmake_minimum_required(VERSION 3.10)
project(ncnn_inference C CXX)

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


# NCNN Path Setting
if(USE_ANDROID)
set(NCNN_INC ${THIRD_PARTY}/ncnn/include/ncnn)
set(NCNN_LIB ${THIRD_PARTY}/ncnn/libs/android_aarch64/libncnn.so)
else()
set(NCNN_INC ${THIRD_PARTY}/ncnn/include/ncnn)
set(NCNN_LIB ${THIRD_PARTY}/ncnn/libs/linux_x64/libncnn.so)
endif()


# OpenCV Path Setting
if(USE_ANDROID)
set(OPENCV_INC ${THIRD_PARTY}/opencv/opencv_android_aarch64/sdk/native/jni/include)
set(OpenCV_LIBS ${THIRD_PARTY}/opencv/opencv_android_aarch64/sdk/native/libs/arm64-v8a)
else()
set(OPENCV_INC ${THIRD_PARTY}/opencv/opencv_linux_x64/include)
set(OpenCV_LIBS ${THIRD_PARTY}/opencv/opencv_linux_x64/libs)
endif()


add_executable(ncnn_inference 
${CMAKE_CURRENT_LIST_DIR}/ncnn_main.cpp
)

# Link NCNN
target_include_directories(ncnn_inference ${ATTRIBUTE} ${NCNN_INC})
target_link_libraries(ncnn_inference ${NCNN_LIB})

# Link OpenCV
target_include_directories(ncnn_inference ${ATTRIBUTE} ${OPENCV_INC})
target_link_libraries(ncnn_inference
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
target_include_directories(ncnn_inference ${ATTRIBUTE} ${OpenCL_INCLUDE_DIRS})
target_link_libraries(ncnn_inference ${OpenCL_LIBRARIES})
endif()

target_link_libraries(ncnn_inference
EGL 
GLESv2
)

if(USE_ANDROID)
target_link_libraries(ncnn_inference GLESv3)
find_library(ANDROID_LOG_LIB log)
target_link_libraries(ncnn_inference log)
endif()
