if (USE_EIGEN)
    INCLUDE(ExternalProject)
    SET(EIGEN_SOURCES_DIR ${THIRD_PARTY_PATH}/eigen)
    SET(EIGEN_INSTALL_DIR ${INSTALL_LIB_PATH}/eigen)
    SET(EIGEN_INCLUDE_DIR "${EIGEN_INSTALL_DIR}/include" CACHE PATH "eigen include directory." FORCE)

    INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

    ADD_DEFINITIONS(-DUSE_EIGEN)

    if (EXISTS ${EIGEN_INSTALL_DIR})
        MESSAGE(STATUS "${EIGEN_INSTALL_DIR} exists")
        add_custom_target(eigen)
        LIST(APPEND external_project_dependencies eigen)
        return()
    endif ()

    ExternalProject_Add(
            eigen
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/eigenteam/eigen-git-mirror.git"
            PREFIX          ${EIGEN_SOURCES_DIR}
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${EIGEN_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies eigen)
endif (USE_EIGEN)