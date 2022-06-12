#!/usr/bin/env bash
mkdir bin_android
cd bin_android

rm CMakeCache.txt
export ANDROID_NDK=/home/s1secom/Downloads/android-ndk-r21e-linux-x86_64/android-ndk-r21e
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
	-DANDROID_ABI="arm64-v8a" \
	-DANDROID_PLATFORM=android-24 \
	-DCMAKE_INSTALL_PREFIX=install \
	-DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_ANDROID_STL_TYPE=gnustl_shared \
	-DUSE_ANDROID=ON \
	..
	
make -j4
