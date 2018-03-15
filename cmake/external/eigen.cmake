if (USE_EIGEN)
    INCLUDE(ExternalProject)
    SET(EIGEN_SOURCES_DIR ${THIRD_PARTY_PATH}/eigen)
    SET(EIGEN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/eigen)
    SET(EIGEN_INCLUDE_DIR "${EIGEN_INSTALL_DIR}/include" CACHE PATH "eigen include directory." FORCE)
    IF(WIN32)
#        SET(EIGEN_LIBRARIES "${EIGEN_INSTALL_DIR}/lib/libeigen.lib")
    else(WIN32)
#        SET(EIGEN_LIBRARIES "${EIGEN_INSTALL_DIR}/lib/libeigen.a")
    endif(WIN32)
    INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})
    ExternalProject_Add(
            eigen
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/eigenteam/eigen-git-mirror.git"
#            UPDATE_COMMAND  git pull
            PREFIX          ${EIGEN_SOURCES_DIR}
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${EIGEN_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -O2"
    )
    LIST(APPEND external_project_dependencies eigen)
#    LIST(APPEND external_libs ${EIGEN_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_EIGEN)
endif (USE_EIGEN)