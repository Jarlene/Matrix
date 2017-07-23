#CUDA
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
    message(STATUS "use mkl")
    set(MKL_INCLUDE_DIR
            /opt/intel/mkl/include)

    set(MKL_LIBS "")
    # runtime libs
    file(GLOB_RECURSE lib_files  /opt/intel/mkl/lib/*.a /opt/intel/mkl/lib/*.dylib)
    foreach(source_file ${lib_files})
        list(APPEND MKL_LIBS ${source_file})
    endforeach()

    # runtime dependence libs
    file(GLOB_RECURSE lib_files  /opt/intel/lib/*.a /opt/intel/lib/*.dylib)
    foreach(source_file ${lib_files})
        list(APPEND MKL_LIBS ${source_file})
    endforeach()

    # runtime dependence libs
    file(GLOB_RECURSE lib_files  /opt/intel/tbb/*.a /opt/intel/tbb/*.dylib)
    foreach(source_file ${lib_files})
        list(APPEND MKL_LIBS ${source_file})
    endforeach()

    include_directories(${MKL_INCLUDE_DIR})
    link_libraries(${MKL_LIBS})
    ADD_DEFINITIONS(-DUSE_MKL)
endif ()




# JNI
#find_package(JNI)
#if (JNI_FOUND)
#    include_directories(${JNI_INCLUDE_DIRS})
#endif()


#OpenCv
if (USE_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
    if (OpenCV_FOUND)
        include_directories(${OpenCV_INCLUDE_DIRS})
        link_libraries(${OpenCV_LIBRARIES})
        add_definitions(-DUSE_OPENCV)
    endif ()

endif ()