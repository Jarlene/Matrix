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
        get_filename_component(FILE_EXT ${DOWNLOAD_OUTPATH} EXT)
        if ("${FILE_EXT}" STREQUAL ".gz")
            execute_process(
                    COMMAND gunzip ${DOWNLOAD_OUTPATH}
            )
        elseif("${FILE_EXT}" STREQUAL ".zip")
            execute_process(
                    COMMAND unzip ${DOWNLOAD_OUTPATH}
            )
        elseif("${FILE_EXT}" STREQUAL ".tar")
            execute_process(
                    COMMAND tar -xvf ${DOWNLOAD_OUTPATH}
            )
        endif ()
    endif ()
endfunction(matrix_download)