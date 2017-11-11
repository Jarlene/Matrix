if(USE_LLVM)
    INCLUDE(ExternalProject)
    SET(LLVM_SOURCES_DIR ${THIRD_PARTY_PATH}/llvm)
    SET(LLVM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/llvm)
    SET(LLVM_INCLUDE_DIR "${LLVM_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

    IF(WIN32)
        SET(LLVM_LIBRARIES "${LLVM_INSTALL_DIR}/lib/libopencv_core.lib" CACHE FILEPATH "opencv library." FORCE)
    ELSE(WIN32)
        SET(LLVM_LIBRARIES "")
        LIST(APPEND LLVM_LIBRARIES   "${LLVM_INSTALL_DIR}/lib/libopencv_core.dylib")
        LIST(APPEND LLVM_LIBRARIES   "${LLVM_INSTALL_DIR}/lib/libopencv_imgcodecs.dylib")
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${LLVM_INCLUDE_DIR})


    ExternalProject_Add(
            llvm
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/llvm-mirror/llvm.git"
            GIT_TAG         "master"
            PREFIX          ${LLVM_SOURCES_DIR}
            UPDATE_COMMAND  ""
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
    )
    LIST(APPEND external_project_dependencies llvm)
    LIST(APPEND external_libs ${LLVM_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_LLVM)
endif(USE_LLVM)
