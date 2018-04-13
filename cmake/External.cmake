set(THIRD_PARTY_PATH "${CMAKE_SOURCE_DIR}/third-party")
set(INSTALL_LIB_PATH "${CMAKE_SOURCE_DIR}/libs")
#include(ExternalProject)
include(external/gtest)
include(external/gflags)
include(external/glog)
include(external/openmp)
include(external/blas)
include(external/eigen)
include(external/opencv)
include(external/llvm)
include(external/zeromq)
include(external/xbuild)