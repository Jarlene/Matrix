#CUDA
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/uitls")
if (USE_CUDA)
    find_package(CUDA)
    if (CUDA_FOUND)
        #    message(STATUS "cude_include = ${CUDA_INCLUDE_DIRS}")
        #    message(STATUS "cude_lib = ${CUDA_LIBRARIES}")
        include_directories(${CUDA_INCLUDE_DIRS})
        link_libraries(${CUDA_LIBRARIES})
        ADD_DEFINITIONS(-DUSE_CUDA)
    endif()
endif ()


# MKL
if (USE_MKL)
    find_package(MKL)
    if (MKL_FOUND)
#        message(STATUS ${MKL_INCLUDE_DIRS})
        include_directories(${MKL_INCLUDE_DIRS})
        link_libraries(${MKL_LIBRARIES})
        ADD_DEFINITIONS(-DUSE_MKL)
    endif ()
endif ()




# JNI
#find_package(JNI)
#if (JNI_FOUND)
#    include_directories(${JNI_INCLUDE_DIRS})
#endif()
