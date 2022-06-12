#!/usr/bin/env bash
mkdir bin_linux
cd bin_linux
rm CMakeCache.txt
cmake 	-DCMAKE_INSTALL_PREFIX=install \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=ON \
	-DOpenCL_LIBRARY="/usr/lib/x86_64-linux-gnu" \
	-DOpenCL_INCLUDE_DIR="/usr/include" \
	..

make -j4
