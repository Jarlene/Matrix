if (USE_BLAS)
    SET(BLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/OpenBLAS)
    SET(BLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/OpenBLAS)
    SET(BLAS_INCLUDE_DIR "${BLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

#    ExternalProject_Add(
#            OpenBLAS
#            ${EXTERNAL_PROJECT_LOG_ARGS}
#            GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
#            GIT_TAG         "master"
#            PREFIX          "${BLAS_SOURCES_DIR}"
#            BUILD_COMMAND   "make"
#            BUILD_IN_SOURCE 1
#            INSTALL_DIR     "${BLAS_INSTALL_DIR}"
#            INSTALL_COMMAND "make install"
#    )

    if (NOT EXISTS ${BLAS_SOURCES_DIR})
        message(STATUS "git clone opeb blas library")
        execute_process(
                COMMAND  git clone https://github.com/xianyi/OpenBLAS.git ${BLAS_SOURCES_DIR}
        )
    endif ()

    if (NOT EXISTS ${BLAS_INSTALL_DIR})
        execute_process(
                WORKING_DIRECTORY ${BLAS_SOURCES_DIR}
                COMMAND make
                COMMAND make install PREFIX=${BLAS_INSTALL_DIR}
        )
    endif ()
    IF(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libopenblas.lib")
    else(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libopenblas.a")
    endif(WIN32)
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIR})
#    LIST(APPEND external_project_dependencies OpenBLAS)
    LIST(APPEND external_libs ${BLAS_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_BLAS)
endif (USE_BLAS)