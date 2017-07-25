if(USE_OPENCV)
    INCLUDE(ExternalProject)
    SET(OPENCV_SOURCES_DIR ${THIRD_PARTY_PATH}/opencv)
    SET(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/opencv)
    SET(OPENCV_INCLUDE_DIR "${OPENCV_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

    IF(WIN32)
        SET(OPENCV_LIBRARIES "${OPENMP_INSTALL_DIR}/lib/libopencv.lib" CACHE FILEPATH "opencv library." FORCE)
    ELSE(WIN32)
        SET(OPENCV_LIBRARIES "${OPENMP_INSTALL_DIR}/lib/libopencv.dylib" CACHE FILEPATH "opencv library." FORCE)
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})


    ExternalProject_Add(
            opencv
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/opencv/opencv.git"
            GIT_TAG         "master"
            PREFIX          ${OPENCV_SOURCES_DIR}
            UPDATE_COMMAND  ""
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR} -DWITH_CUDA=OFF -DWITH_CUFFT=OFF -DWITH_CUBLAS=OFF -DWITH_NVCUVID=OFF
    )
    LIST(APPEND external_project_dependencies opencv)
    LIST(APPEND external_libs ${OPENCV_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_OPENCV)
endif(USE_OPENCV)
