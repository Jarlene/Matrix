IF(USE_TEST)
    ENABLE_TESTING()
    INCLUDE(ExternalProject)

    SET(GTEST_SOURCES_DIR ${THIRD_PARTY_PATH}/gtest)
    SET(GTEST_INSTALL_DIR ${THIRD_PARTY_PATH}/install/gtest)
    SET(GTEST_INCLUDE_DIR "${GTEST_INSTALL_DIR}/include" CACHE PATH "gtest include directory." FORCE)

    INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIR})

    IF(WIN32)
        set(GTEST_LIBRARIES
            "${GTEST_INSTALL_DIR}/lib/gtest.lib" CACHE FILEPATH "gtest libraries." FORCE)
        set(GTEST_MAIN_LIBRARIES
            "${GTEST_INSTALL_DIR}/lib/gtest_main.lib" CACHE FILEPATH "gtest main libraries." FORCE)
    ELSE(WIN32)
        set(GTEST_LIBRARIES
            "${GTEST_INSTALL_DIR}/lib/libgtest.a" CACHE FILEPATH "gtest libraries." FORCE)
        set(GTEST_MAIN_LIBRARIES
            "${GTEST_INSTALL_DIR}/lib/libgtest_main.a" CACHE FILEPATH "gtest main libraries." FORCE)
    ENDIF(WIN32)

    ExternalProject_Add(
        gtest
        ${EXTERNAL_PROJECT_LOG_ARGS}
        GIT_REPOSITORY  "https://github.com/google/googletest.git"
        PREFIX          ${GTEST_SOURCES_DIR}
#        UPDATE_COMMAND  "git" "pull"
        CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${GTEST_INSTALL_DIR}
        CMAKE_ARGS      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        CMAKE_ARGS      -DBUILD_GMOCK=ON
        CMAKE_ARGS      -Dgtest_disable_pthreads=ON
        CMAKE_ARGS      -Dgtest_force_shared_crt=ON
    )
    LIST(APPEND external_project_dependencies gtest)
    LIST(APPEND external_libs ${GTEST_LIBRARIES})
ENDIF(USE_TEST)
