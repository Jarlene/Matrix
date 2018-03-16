if (USE_OPENMP)
    INCLUDE(ExternalProject)
    SET(OPENMP_SOURCES_DIR ${THIRD_PARTY_PATH}/openmp)
    SET(OPENMP_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openmp)
    SET(OPENMP_INCLUDE_DIR "${OPENMP_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

    IF(WIN32)
        SET(OPENMP_LIBRARIES "${OPENMP_INSTALL_DIR}/lib/libomp.lib" CACHE FILEPATH "openmp library." FORCE)
    ELSE(WIN32)
        SET(OPENMP_LIBRARIES "${OPENMP_INSTALL_DIR}/lib/libomp.dylib" CACHE FILEPATH "openmp library." FORCE)
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${OPENMP_INCLUDE_DIR})


    ExternalProject_Add(
            openmp
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/llvm-mirror/openmp.git"
            PREFIX          ${OPENMP_SOURCES_DIR}
#            UPDATE_COMMAND  "git" "pull"
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${OPENMP_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS="-O2"
    )
    LIST(APPEND external_project_dependencies openmp)
    LIST(APPEND external_libs ${OPENMP_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_MP)
endif (USE_OPENMP)