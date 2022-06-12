#!/usr/bin/env bash
mkdir bin_android
cd bin_android

rm CMakeCache.txt
export ANDROID_NDK=../../0_3rdparty/android-ndk-r21e-linux-x86_64
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake"\
	-DCMAKE_SYSTEM_NAME=Android \
 	-DCMAKE_SYSTEM_VERSION=21 \
	-DANDROID_ABI="arm64-v8a" \
	-DANDROID_PLATFORM=android-24 \
	-DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
	-DCMAKE_INSTALL_PREFIX=install \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=ON \
	-DCMAKE_ANDROID_NDK=$ANDROID_NDK \
	-DCMAKE_ANDROID_STL_TYPE=gnustl_static \
	-DOpenCL_LIBRARY="/usr/lib/x86_64-linux-gnu" \
	-DOpenCL_INCLUDE_DIR="/usr/include" \
	-DUSE_ANDROID=ON \
	..
	
make -j4

