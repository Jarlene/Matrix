if (USE_BLAS)
    INCLUDE(ExternalProject)
    SET(BLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/OpenBLAS)
    SET(BLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/OpenBLAS)
    SET(BLAS_INCLUDE_DIR "${BLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)
    IF(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libOpenBLAS.lib")
    else(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libOpenBLAS.a")
    endif(WIN32)

    ExternalProject_Add(
            OpenBLAS
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
            GIT_TAG  "master"
            PREFIX          ${BLAS_SOURCES_DIR}
            UPDATE_COMMAND  ""
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${BLAS_INSTALL_DIR}
    )
    LIST(APPEND external_project_dependencies OpenBLAS)
    LIST(APPEND external_libs ${BLAS_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_BLAS)
endif (USE_BLAS)