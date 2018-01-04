if(USE_GFLAGS)
    INCLUDE(ExternalProject)

    SET(GFLAGS_SOURCES_DIR ${THIRD_PARTY_PATH}/gflags)
    SET(GFLAGS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gflags)
    SET(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
    IF(WIN32)
        set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/gflags.lib" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
    ELSE(WIN32)
        set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIR})

    ExternalProject_Add(
            gflags
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/gflags/gflags.git"
            PREFIX          ${GFLAGS_SOURCES_DIR}
#            UPDATE_COMMAND  "git" "pull"
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            CMAKE_ARGS      -DBUILD_TESTING=OFF
    )
    LIST(APPEND external_project_dependencies gflags)
    LIST(APPEND external_libs ${GFLAGS_LIBRARIES})
ENDIF(USE_GFLAGS)