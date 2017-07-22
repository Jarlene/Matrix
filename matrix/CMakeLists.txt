
set(headList "")
file(GLOB_RECURSE header_files include/*.h include/*.hpp)
foreach(source_file ${header_files})
    list(APPEND headList ${source_file})
endforeach()

# src files
set(srcList "")
file(GLOB_RECURSE header_files src/*.cpp src/*.cc)
foreach(source_file ${header_files})
    list(APPEND srcList ${source_file})
endforeach()

set(SOURCE_FILES ${headList} ${srcList})
add_library(Matrix STATIC ${SOURCE_FILES})