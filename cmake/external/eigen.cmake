if (USE_EIGEN)
    INCLUDE(ExternalProject)
    SET(EIGEN_SOURCES_DIR ${THIRD_PARTY_PATH}/eigen)
    SET(EIGEN_INSTALL_DIR ${THIRD_PARTY_PATH}/install/eigen)
    SET(EIGEN_INCLUDE_DIR "${EIGEN_INSTALL_DIR}/include" CACHE PATH "eigen include directory." FORCE)
    IF(WIN32)
        SET(EIGEN_LIBRARIES "${EIGEN_INSTALL_DIR}/lib/libeigen.lib")
    else(WIN32)
        SET(EIGEN_LIBRARIES "${EIGEN_INSTALL_DIR}/lib/libeigen.a")
    endif(WIN32)

    ExternalProject_Add(
            eigen
            ${EXTERNAL_PROJECT_LOG_ARGS}
            HG_REPOSITORY "https://bitbucket.org/eigen/eigen/"
            #            HG_TAG  ""
            PREFIX          ${EIGEN_SOURCES_DIR}
            UPDATE_COMMAND  ""
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${EIGEN_INSTALL_DIR}
    )
    LIST(APPEND external_project_dependencies eigen)
    LIST(APPEND external_libs ${EIGEN_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_EIGEN)
endif (USE_EIGEN)