if (USE_XBUILD)
    INCLUDE(ExternalProject)
    SET(XBUILD_SOURCES_DIR ${THIRD_PARTY_PATH}/xbuild)
    SET(XBUILD_INSTALL_DIR ${INSTALL_LIB_PATH}/xbuild)
    SET(XBUILD_INCLUDE_DIR "${XBUILD_INSTALL_DIR}/include" CACHE PATH "xbuild include directory." FORCE)
    IF (WIN32)
        SET(XBUILD_LIBRARIES "${XBUILD_INSTALL_DIR}/lib/xbuild.lib")
    else (WIN32)
        SET(XBUILD_LIBRARIES "${XBUILD_INSTALL_DIR}/lib/libxbuild.so")
    endif (WIN32)
    INCLUDE_DIRECTORIES(${XBUILD_INCLUDE_DIR})


    LIST(APPEND external_libs ${XBUILD_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_XBUILD)

    if (EXISTS ${XBUILD_INSTALL_DIR})
        MESSAGE(STATUS "${XBUILD_INSTALL_DIR} exists")
        return()
    endif ()

    ExternalProject_Add(
            xbuild
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/Jarlene/XBuild.git"
            PREFIX ${XBUILD_SOURCES_DIR}
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${XBUILD_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies xbuild)
endif (USE_XBUILD)