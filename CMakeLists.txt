cmake_minimum_required(VERSION 3.7)
project(Matrix)

set(CMAKE_CXX_STANDARD 14)

option(USE_TEST "use gtest" ON)
option(USE_GLOG "use glog" ON)
option(USE_GFLAGS "use gflags" ON)
option(USE_OPENMP "use openmp" ON)
option(USE_BLAS "use blas" ON)
option(USE_EIGEN "use eigen" ON)

include_directories("${CMAKE_SOURCE_DIR}")
set(THIRD_PARTY_PATH "${CMAKE_SOURCE_DIR}/third-party")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(ExternalProject)

include(System)
include(Dependencies)
include(external/gtest)
include(external/gflags)
include(external/glog)
include(external/openmp)
include(external/blas)
include(external/eigen)



add_subdirectory(matrix)
add_subdirectory(sample)


