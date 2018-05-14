if(USE_OPENCV)
    INCLUDE(ExternalProject)
    SET(OPENCV_SOURCES_DIR ${THIRD_PARTY_PATH}/opencv)
    SET(OPENCV_INSTALL_DIR ${INSTALL_LIB_PATH}/opencv)
    SET(OPENCV_INCLUDE_DIR "${OPENCV_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

    IF(WIN32)
        SET(OPENCV_LIBRARIES "${OPENCV_INSTALL_DIR}/lib/libopencv_core.lib" CACHE FILEPATH "opencv library." FORCE)
    ELSE(WIN32)
        SET(OPENCV_LIBRARIES "")
        LIST(APPEND OPENCV_LIBRARIES   "${OPENCV_INSTALL_DIR}/lib/libopencv_core.dylib")
        LIST(APPEND OPENCV_LIBRARIES   "${OPENCV_INSTALL_DIR}/lib/libopencv_core.4.0.dylib")
        LIST(APPEND OPENCV_LIBRARIES   "${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.dylib")
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})

    LIST(APPEND external_libs ${OPENCV_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_OPENCV)

    if (EXISTS ${OPENCV_INSTALL_DIR})
        MESSAGE(STATUS "${OPENCV_INSTALL_DIR} exists")
        add_custom_target(opencv)
        LIST(APPEND external_project_dependencies opencv)
        return()
    endif ()

    ExternalProject_Add(
            opencv
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/opencv/opencv.git"
            GIT_TAG         "master"
            PREFIX          ${OPENCV_SOURCES_DIR}
#            UPDATE_COMMAND  "git" "pull"
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
            CMAKE_ARGS      -DWITH_CUDA=OFF
            CMAKE_ARGS      -DWITH_CUFFT=OFF
            CMAKE_ARGS      -DWITH_CUBLAS=OFF
            CMAKE_ARGS      -DWITH_NVCUVID=OFF
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies opencv)
endif(USE_OPENCV)
