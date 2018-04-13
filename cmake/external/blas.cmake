if (USE_BLAS)
    SET(BLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/openblas)
    SET(BLAS_INSTALL_DIR ${INSTALL_LIB_PATH}/openblas)
    SET(BLAS_INCLUDE_DIR "${BLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

    find_package(OpenBLAS)

    if (OpenBLAS_FOUND)
        MESSAGE(STATUS ${OpenBLAS_INCLUDE_DIRS})
        INCLUDE_DIRECTORIES(${OpenBLAS_INCLUDE_DIRS})
        LIST(APPEND external_libs ${OpenBLAS_LIBRARIES})
        ADD_DEFINITIONS(-DUSE_BLAS)
        return()
    endif ()


    IF(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libopenblas.lib")
    else(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libopenblas.a")
    endif(WIN32)
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIR})

    LIST(APPEND external_libs ${BLAS_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_BLAS)
    if (EXISTS ${BLAS_INSTALL_DIR})
        MESSAGE(STATUS "${BLAS_INSTALL_DIR} exists")
        add_custom_target(openblas)
        LIST(APPEND external_project_dependencies openblas)
        return()
    endif ()

    ExternalProject_Add(
            openblas
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
            GIT_TAG         "master"
            PREFIX          "${BLAS_SOURCES_DIR}"
            CMAKE_ARGS      -DCMAKE_INSTALL_INCLUDEDIR=${BLAS_INCLUDE_DIR}
            CMAKE_ARGS      -DCMAKE_INSTALL_LIBDIR=${BLAS_INSTALL_DIR}/lib
            CMAKE_ARGS      -DCMAKE_INSTALL_BINDIR=${BLAS_INSTALL_DIR}/bin
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies openblas)

endif (USE_BLAS)