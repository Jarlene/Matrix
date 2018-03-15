if(USE_LLVM)

    find_package(LLVM)
    if (LLVM_FOUND)
        MESSAGE(STATUS ${LLVM_INCLUDE_DIRS})
        INCLUDE_DIRECTORIES(${LLVM_INCLUDE_DIRS})
        LIST(APPEND external_libs ${LLVM_LIBRARIES})
        ADD_DEFINITIONS(-DUSE_LLVM)
        return()
    endif (LLVM_FOUND)


    INCLUDE(ExternalProject)
    SET(LLVM_SOURCES_DIR ${THIRD_PARTY_PATH}/llvm)
    SET(LLVM_INSTALL_DIR ${THIRD_PARTY_PATH}/install/llvm)
    SET(LLVM_INCLUDE_DIR "${LLVM_INSTALL_DIR}/include" CACHE PATH "glog include directory." FORCE)

    IF(WIN32)
        SET(LLVM_LIBRARIES "${LLVM_INSTALL_DIR}/lib/*.a" CACHE FILEPATH "llvm library." FORCE)
    ELSE(WIN32)
        SET(LLVM_LIBRARIES "")
        LIST(APPEND LLVM_LIBRARIES   "${LLVM_INSTALL_DIR}/lib/*.dylib")
        LIST(APPEND LLVM_LIBRARIES   "${LLVM_INSTALL_DIR}/lib/*.dylib")
    ENDIF(WIN32)

    INCLUDE_DIRECTORIES(${LLVM_INCLUDE_DIR})


    ExternalProject_Add(
            llvm
            ${EXTERNAL_PROJECT_LOG_ARGS}
            GIT_REPOSITORY  "https://github.com/llvm-mirror/llvm.git"
            GIT_TAG         "master"
            PREFIX          ${LLVM_SOURCES_DIR}
#            UPDATE_COMMAND  "git" "pull"
            CMAKE_ARGS      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}
            CMAKE_ARGS      -DCMAKE_BUILD_TYPE=Release
            CMAKE_ARGS      -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -O2"
    )
    LIST(APPEND external_project_dependencies llvm)
    LIST(APPEND external_libs ${LLVM_LIBRARIES})
    ADD_DEFINITIONS(-DUSE_LLVM)
endif(USE_LLVM)
