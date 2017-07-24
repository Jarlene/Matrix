cmake_minimum_required(VERSION 3.7)
project(Matrix)

set(CMAKE_CXX_STANDARD 14)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
include(Option)
include(System)
include(Dependencies)
include(Download)
include(External)


include_directories("${CMAKE_SOURCE_DIR}")

add_subdirectory(matrix)
add_subdirectory(sample)
add_subdirectory(test)

