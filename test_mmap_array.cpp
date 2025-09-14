// test_mmap_array.cpp
#include "cnpy.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <string>
#include <vector>

TEST_CASE("MMap-backed NpyArray lifecycle and persistence", "[cnpy][mmap]") {
    const std::string filename = "test_mmap_array.npy";
    std::vector<size_t> shape = {2, 3};

    // Create a mmap-backed array and write initial data
    {
        cnpy::NpyArray arr = cnpy::new_mmap<int>(filename, shape, false);
        int* data = arr.data<int>();
        size_t total = arr.num_vals;
        for (size_t i = 0; i < total; ++i) {
            data[i] = static_cast<int>(i * 10);
        }
        // arr goes out of scope, unmapping the file
    }

    // Load the file without mmap and verify the data
    cnpy::NpyArray arr_load = cnpy::npy_load(filename, false);
    const int* loaded_data = arr_load.data<int>();
    size_t total = arr_load.num_vals;
    for (size_t i = 0; i < total; ++i) {
        REQUIRE(loaded_data[i] == static_cast<int>(i * 10));
    }

    // Modify the data and save it back to the file
    int* mod_data = arr_load.data<int>();
    for (size_t i = 0; i < total; ++i) {
        mod_data[i] = static_cast<int>(i * 100);
    }
    cnpy::npy_save<int>(filename, mod_data, shape, "w");

    // Load the file with mmap and verify the modifications persisted
    cnpy::NpyArray arr_mmap = cnpy::npy_load(filename, true);
    const int* mmap_data = arr_mmap.data<int>();
    for (size_t i = 0; i < total; ++i) {
        REQUIRE(mmap_data[i] == static_cast<int>(i * 100));
    }

    // Clean up the test file
    std::remove(filename.c_str());
}
// Test for npy_save and npy_load with multi-dimensional data
TEST_CASE("npy_save and npy_load with multi-dimensional data", "[cnpy]") {
    // Prepare data for a 2x3x4 array (24 elements)
    std::vector<int> data(2 * 3 * 4);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i);
    }
    std::vector<size_t> shape = {2, 3, 4};
    std::string filename = "test_npy_load_multi.npy";
    // Save using the pointer overload of npy_save
    cnpy::npy_save<int>(filename, data.data(), shape);
    // Load the file using npy_load
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Verify shape and metadata
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(int));
    // Verify the loaded data matches the original
    const int* loaded = arr.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    // Clean up the temporary .npy file
    std::remove(filename.c_str());
}
// Test for npy_save and npy_load with long type data
TEST_CASE("npy_save and npy_load with long type data", "[cnpy]") {
    std::vector<long> data = {10, 20, 30, 40, 50};
    std::string filename = "test_npy_load_long.npy";
    // Save the data to a .npy file using the library's npy_save overload for vectors
    cnpy::npy_save<long>(filename, data);
    // Load the .npy file back into an NpyArray
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Verify shape and metadata
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(long));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(long));
    // Verify the loaded data matches the original
    const long* loaded = arr.data<long>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    // Clean up the temporary .npy file
    std::remove(filename.c_str());
}
// Tests cnpy::npy_save/load for double type data
TEST_CASE("npy_save and npy_load with double type data", "[cnpy]") {
    std::vector<double> data = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::string filename = "test_npy_load_double.npy";
    // Save the data to a .npy file using the library's npy_save overload for vectors
    cnpy::npy_save<double>(filename, data);
    // Load the .npy file back into an NpyArray
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Verify shape and metadata
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(double));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(double));
    // Verify the loaded data matches the original
    const double* loaded = arr.data<double>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == Catch::Approx(data[i]));
    }
    // Clean up the temporary .npy file
    std::remove(filename.c_str());
}
// Tests cnpy::npy_save/load for long double type
TEST_CASE("npy_save and npy_load with long double type data", "[cnpy]") {
    std::vector<long double> data = {0.1L, 0.2L, 0.3L, 0.4L, 0.5L};
    std::string filename = "test_npy_load_long_double.npy";
    // Save the data to a .npy file using the library's npy_save overload for vectors
    cnpy::npy_save<long double>(filename, data);
    // Load the .npy file back into an NpyArray
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Verify shape and metadata
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(long double));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(long double));
    // Verify the loaded data matches the original
    const long double* loaded = arr.data<long double>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == Catch::Approx(data[i]));
    }
    // Clean up the temporary .npy file
    std::remove(filename.c_str());
}