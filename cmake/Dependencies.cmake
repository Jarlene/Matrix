# OpenBlas
if (USE_BLAS)
    SET(Open_BLAS_INCLUDE_SEARCH_PATHS
            /usr/include
            /usr/include/openblas
            /usr/include/openblas-base
            /usr/local/include
            /usr/local/include/openblas
            /usr/local/include/openblas-base
            /usr/local/OpenBLAS/include
            /usr/local/opt/openblas/include
            )

    SET(Open_BLAS_LIB_SEARCH_PATHS
            /lib/
            /lib/openblas-base
            /lib64/
            /usr/lib
            /usr/lib/openblas-base
            /usr/lib64
            /usr/local/lib
            /usr/local/lib64
            /usr/local/OpenBLAS/lib
            /usr/local/opt/openblas/lib
            )

    FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
    FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

    include_directories(${OpenBLAS_INCLUDE_DIR})
    link_libraries(${OpenBLAS_LIB})
    ADD_DEFINITIONS(-DUSE_BLAS)
endif ()

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

#OpenMp
if (USE_MP)
    FIND_PACKAGE(OpenMP REQUIRED)
    if(OPENMP_FOUND)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
        ADD_DEFINITIONS(-DUSE_MP)
    endif()
endif ()


# JNI
#find_package(JNI)
#if (JNI_FOUND)
#    include_directories(${JNI_INCLUDE_DIRS})
#endif()

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
    add_subdirectory(${CMAKE_SOURCE_DIR}/cuda)
endif ()

#OpenCv
if (USE_OPENCV)
    find_package(OpenCV QUIET COMPONENTS core highgui imgproc imgcodecs)
    if (OpenCV_FOUND)
        include_directories(${OpenCV_INCLUDE_DIRS})
        link_libraries(${OpenCV_LIBRARIES})
        add_definitions(-DUSE_OPENCV)
    endif ()

endif ()