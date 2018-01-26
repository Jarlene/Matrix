
configure_file(
        "${PROJECT_SOURCE_DIR}/matrix/Version.h.in"
        "${PROJECT_BINARY_DIR}/Version.h"
)

set(headList "")
file(GLOB_RECURSE header_files include/*.h include/*.hpp)
foreach(source_file ${header_files})
    list(APPEND headList ${source_file})
endforeach()

# src files
set(srcList "")

if (USE_CUDA)
    file(GLOB_RECURSE src_files src/*.cu)
else()
    file(GLOB_RECURSE src_files src/*.cpp src/*.cc)
endif ()

foreach(source_file ${src_files})
    list(APPEND srcList ${source_file})
endforeach()

set(SOURCE_FILES ${headList} ${srcList})
add_library(Matrix SHARED ${SOURCE_FILES})
SET_TARGET_PROPERTIES(Matrix PROPERTIES OUTPUT_NAME "Matrix" CLEAN_DIRECT_OUTPUT 1 SUFFIX ".so")
add_dependencies(Matrix ${external_project_dependencies})
target_link_libraries(Matrix ${external_libs})