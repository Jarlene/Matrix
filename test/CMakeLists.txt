if (USE_TEST)
    file(GLOB_RECURSE examples_srcs "${PROJECT_SOURCE_DIR}/test/*.cpp")

    foreach(source_file ${examples_srcs})
        # get file name
        get_filename_component(name ${source_file} NAME_WE)

        # get folder name
        get_filename_component(path ${source_file} PATH)
        get_filename_component(folder ${path} NAME_WE)

        add_executable(${name} ${source_file}  include/Test.h)
        target_link_libraries(${name}  Matrix ${GTEST_MAIN_LIBRARIES})
        add_test(NAME ${name} COMMAND ${name})
    endforeach()
endif (USE_TEST)