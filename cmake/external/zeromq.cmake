if (USE_ZMQ)
    INCLUDE(ExternalProject)
    SET(ZMQ_SOURCES_DIR ${THIRD_PARTY_PATH}/zmq)
    SET(ZMQ_INSTALL_DIR ${INSTALL_LIB_PATH}/zmq)
    SET(ZMQ_INCLUDE_DIR "${ZMQ_INSTALL_DIR}/include" CACHE PATH "zmp include directory." FORCE)
    IF (WIN32)
        SET(ZMQ_LIBRARIES "${ZMQ_INSTALL_DIR}/lib/libzmq.lib")
    else (WIN32)
        SET(ZMQ_LIBRARIES "${ZMQ_INSTALL_DIR}/lib/libzmq.a")
    endif (WIN32)
    INCLUDE_DIRECTORIES(${ZMQ_INCLUDE_DIR})



    LIST(APPEND external_libs ${ZMQ_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_ZMQ)

    if (EXISTS ${ZMQ_INSTALL_DIR})
        MESSAGE(STATUS "${ZMQ_INSTALL_DIR} exists")
        add_custom_target(zmq)
        LIST(APPEND external_project_dependencies zmq)
        return()
    endif ()

    ExternalProject_Add(
            zmq
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/zeromq/libzmq.git"
            PREFIX ${ZMQ_SOURCES_DIR}
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX:PATH=${ZMQ_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS=-O2
    )
    LIST(APPEND external_project_dependencies zmq)

endif (USE_ZMQ)