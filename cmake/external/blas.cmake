if (USE_BLAS)
    SET(BLAS_SOURCES_DIR ${THIRD_PARTY_PATH}/openblas)
    SET(BLAS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/openblas)
    SET(BLAS_INCLUDE_DIR "${BLAS_INSTALL_DIR}/include" CACHE PATH "openblas include directory." FORCE)

    find_package(OpenBLAS)

    if (OpenBLAS_FOUND)
        MESSAGE(STATUS ${OpenBLAS_INCLUDE_DIRS})
        INCLUDE_DIRECTORIES(${OpenBLAS_INCLUDE_DIRS})
        LIST(APPEND external_libs ${OpenBLAS_LIBRARIES})
        return()
    endif ()

    ExternalProject_Add(
            openblas
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY "https://github.com/xianyi/OpenBLAS.git"
            GIT_TAG         "master"
            UPDATE_COMMAND  git pull
            PREFIX          "${BLAS_SOURCES_DIR}"
            CMAKE_ARGS      -DCMAKE_INSTALL_INCLUDEDIR=${BLAS_INCLUDE_DIR} -DCMAKE_INSTALL_LIBDIR=${BLAS_INSTALL_DIR}/lib -DCMAKE_INSTALL_BINDIR=${BLAS_INSTALL_DIR}/bin
    )

#    SET(error_code 1)
#    if (NOT EXISTS ${BLAS_SOURCES_DIR})
#        message(STATUS "git clone open blas library")
#        execute_process(
#                COMMAND  git clone -b master https://github.com/xianyi/OpenBLAS.git ${BLAS_SOURCES_DIR}
#                RESULT_VARIABLE error_code
#        )
#    else()
#        set(ouput_msg "")
#        execute_process(
#                COMMAND git pull
#                OUTPUT_VARIABLE ouput_msg
#        )
#        string(FIND "${ouput_msg}" "up-to-date." error_code)
#    endif ()
#
#    if (NOT error_code)
#        MESSAGE(WARNING "git clone fail")
#        return()
#    endif ()
#    if (NOT EXISTS ${BLAS_INSTALL_DIR})
#        SET(error_code 1)
#        execute_process(
#                COMMAND make
#                WORKING_DIRECTORY ${BLAS_SOURCES_DIR}
#                RESULT_VARIABLE error_code
#        )
#        if(NOT error_code)
#            execute_process(
#                    COMMAND make install PREFIX=${BLAS_INSTALL_DIR}
#                    WORKING_DIRECTORY ${BLAS_SOURCES_DIR}
#            )
#        endif()
#    endif ()
    IF(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libopenblas.lib")
    else(WIN32)
        SET(BLAS_LIBRARIES "${BLAS_INSTALL_DIR}/lib/libopenblas.a")
    endif(WIN32)
    INCLUDE_DIRECTORIES(${BLAS_INCLUDE_DIR})
#    LIST(APPEND external_project_dependencies OpenBLAS)
    LIST(APPEND external_libs ${BLAS_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_BLAS)
endif (USE_BLAS)