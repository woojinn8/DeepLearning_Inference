#!/usr/bin/env bash
mkdir bin_linux
cd bin_linux
rm CMakeCache.txt
cmake 	-DCMAKE_INSTALL_PREFIX=install \
	-DCMAKE_BUILD_TYPE=Release \
	..

make -j4
