IF(USE_TEST)


    find_package(GTEST)
    if (GTEST_FOUND)
        MESSAGE(STATUS ${GTEST_INCLUDE_DIRS})
        INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})
        LIST(APPEND external_libs ${GTEST_LIBRARIES})
        ADD_DEFINITIONS(-DUSE_TEST)
        return()
    endif ()

    ENABLE_TESTING()
    INCLUDE(ExternalProject)

    SET(GTEST_SOURCES_DIR ${THIRD_PARTY_PATH}/gtest)
    SET(GTEST_INSTALL_DIR ${INSTALL_LIB_PATH}/gtest)
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


    LIST(APPEND external_libs ${GTEST_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_TEST)


    if (EXISTS ${GTEST_INSTALL_DIR})
        MESSAGE(STATUS "${GTEST_INSTALL_DIR} exists")
        add_custom_target(gtest)
        LIST(APPEND external_project_dependencies gtest)
        return()
    endif ()

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
        CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
        CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies gtest)
ENDIF(USE_TEST)
