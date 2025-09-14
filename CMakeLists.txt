CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
if(COMMAND cmake_policy)
	cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(CNPY)
set(CMAKE_CXX_STANDARD 11)

option(ENABLE_STATIC "Build static (.a) library" ON)

find_package(ZLIB REQUIRED)

include_directories(${ZLIB_INCLUDE_DIRS})

# Add Catch2 testing framework
include(FetchContent)

FetchContent_Declare(
		catch
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.5.0
)

FetchContent_MakeAvailable(catch)
add_library(cnpy SHARED "cnpy.cpp")
target_link_libraries(cnpy ${ZLIB_LIBRARIES})
install(TARGETS "cnpy" LIBRARY DESTINATION lib PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

if(ENABLE_STATIC)
    add_library(cnpy-static STATIC "cnpy.cpp")
    set_target_properties(cnpy-static PROPERTIES OUTPUT_NAME "cnpy")
    install(TARGETS "cnpy-static" ARCHIVE DESTINATION lib)
endif(ENABLE_STATIC)

install(FILES "cnpy.h" DESTINATION include)
install(FILES "mat2npz" "npy2mat" "npz2mat" DESTINATION bin PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

add_executable(example1 example1.cpp)
target_link_libraries(example1 cnpy)

# Enable testing
enable_testing()

add_executable(test_cnpy test_cnpy.cpp)
target_link_libraries(test_cnpy PRIVATE cnpy Catch2::Catch2WithMain)

add_test(NAME cnpy_test COMMAND test_cnpy)

add_executable(test_mmap_util test_mmap_util.cpp)
target_link_libraries(test_mmap_util PRIVATE cnpy Catch2::Catch2WithMain)
add_test(NAME mmap_util_test COMMAND test_mmap_util)
add_executable(test_mmap_array test_mmap_array.cpp)
target_link_libraries(test_mmap_array PRIVATE cnpy Catch2::Catch2WithMain)
add_test(NAME mmap_array_test COMMAND test_mmap_array)
add_executable(test_mmap_save_load test_mmap_save_load.cpp)
target_link_libraries(test_mmap_save_load PRIVATE cnpy Catch2::Catch2WithMain)
add_test(NAME mmap_save_load_test COMMAND test_mmap_save_load)

add_executable(test_generic_regression test_generic_regression.cpp)
target_link_libraries(test_generic_regression PRIVATE cnpy Catch2::Catch2WithMain)
add_test(NAME generic_regression_test COMMAND test_generic_regression)
