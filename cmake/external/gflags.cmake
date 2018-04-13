if(USE_GFLAGS)
    INCLUDE(ExternalProject)

    SET(GFLAGS_SOURCES_DIR ${THIRD_PARTY_PATH}/gflags)
    SET(GFLAGS_INSTALL_DIR ${INSTALL_LIB_PATH}/gflags)
    SET(GFLAGS_INCLUDE_DIR "${GFLAGS_INSTALL_DIR}/include" CACHE PATH "gflags include directory." FORCE)
    IF(WIN32)
        set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/gflags.lib" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
    ELSE(WIN32)
        set(GFLAGS_LIBRARIES "${GFLAGS_INSTALL_DIR}/lib/libgflags.a" CACHE FILEPATH "GFLAGS_LIBRARIES" FORCE)
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${GFLAGS_INCLUDE_DIR})

    LIST(APPEND external_libs ${GFLAGS_LIBRARIES})

    if (EXISTS ${GFLAGS_INSTALL_DIR})
        MESSAGE(STATUS "${GFLAGS_INSTALL_DIR} exists")
        add_custom_target(gflags)
        LIST(APPEND external_project_dependencies gflags)
        return()
    endif ()

    ExternalProject_Add(
            gflags
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/gflags/gflags.git"
            PREFIX          ${GFLAGS_SOURCES_DIR}
#            UPDATE_COMMAND  "git" "pull"
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${GFLAGS_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            CMAKE_ARGS      -DBUILD_TESTING=OFF
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies gflags)
ENDIF(USE_GFLAGS)