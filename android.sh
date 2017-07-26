#!/usr/bin/env bash
mkdir -p android-build
cd android-build
cmake -DCMAKE_TOOLCHAIN_FILE=../cmake/android/android.toolchain.cmake \
      -DANDROID_STL=stlport_shared \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_NATIVE_API_LEVEL=android-24 \
      -DANDROID_TOOLCHAIN_NAME=aarch64-linux-android-4.9 \
      -DANDROID_NDK=/Users/jarlene/Library/Android/sdk/ndk-bundle  ..
make
make install
