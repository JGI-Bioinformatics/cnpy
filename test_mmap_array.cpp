// test_mmap_array.cpp
#include "cnpy.h"
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