file(GLOB_RECURSE examples_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")


foreach(source_file ${examples_srcs})
    # get file name
    get_filename_component(name ${source_file} NAME_WE)

    # get folder name
    get_filename_component(path ${source_file} PATH)
    get_filename_component(folder ${path} NAME_WE)

    add_executable(${name} ${source_file})
    target_link_libraries(${name} Matrix)
endforeach()