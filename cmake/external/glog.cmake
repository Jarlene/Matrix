if (USE_GLOG)
    INCLUDE(ExternalProject)

    SET(GLOG_SOURCES_DIR ${THIRD_PARTY_PATH}/glog)
    SET(GLOG_INSTALL_DIR ${THIRD_PARTY_PATH}/install/glog)
    SET(GLOG_INCLUDE_DIR "${GLOG_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

    IF(WIN32)
        SET(GLOG_LIBRARIES "${GLOG_INSTALL_DIR}/lib/libglog.lib" CACHE FILEPATH "glog library." FORCE)
    ELSE(WIN32)
        SET(GLOG_LIBRARIES "${GLOG_INSTALL_DIR}/lib/libglog.a" CACHE FILEPATH "glog library." FORCE)
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${GLOG_INCLUDE_DIR})

    ExternalProject_Add(
            glog
            ${EXTERNAL_PROJECT_LOG_ARGS}
            DEPENDS gflags
            GIT_REPOSITORY  "https://github.com/google/glog.git"
            PREFIX          ${GLOG_SOURCES_DIR}
#            UPDATE_COMMAND  "git" "pull"
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${GLOG_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            CMAKE_ARGS      -DWITH_GFLAGS=ON
            CMAKE_ARGS      -Dgflags_DIR=${GFLAGS_INSTALL_DIR}/lib/cmake/gflags
            CMAKE_ARGS      -DBUILD_TESTING=OFF
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -O2"
    )
    LIST(APPEND external_project_dependencies glog)
    LIST(APPEND external_libs ${GLOG_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_GLOG)
endif (USE_GLOG)
