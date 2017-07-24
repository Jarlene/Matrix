function(matrix_download)
    cmake_parse_arguments(DOWNLOAD "" "URL;OUTPATH"  "ARGS"  ${ARGN})
    if ("${DOWNLOAD_URL}" STREQUAL "")
        MESSAGE(STATUS "DOWNLOAD_URL is null")
        return()
    endif ()

    if ("${DOWNLOAD_OUTPATH}" STREQUAL "")
        MESSAGE(STATUS "DOWNLOAD_OUTPATH is null")
        return()
    endif ()

    if (NOT EXISTS ${DOWNLOAD_OUTPATH})
        FILE(DOWNLOAD ${DOWNLOAD_URL} ${DOWNLOAD_OUTPATH} SHOW_PROGRESS)
    endif ()
endfunction(matrix_download)