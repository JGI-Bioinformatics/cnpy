// test_mmap_save_load.cpp
#include "cnpy.h"
#include "mmap_util.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <string>
#include <vector>

TEST_CASE("NpyArray save, mmap load, modify, and reload", "[cnpy][mmap]") {
    const std::string filename = "test_mmap_save_load.npy";
    std::vector<size_t> shape = {2, 3};

    // Create initial data and save using npy_save (non-mmap)
    std::vector<int> data(shape[0] * shape[1]);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<int>(i * 10);
    }
    cnpy::npy_save<int>(filename, data.data(), shape, "w");

    // Load the file as read-write mmap and verify initial values
    {
        cnpy::NpyArray arr_mmap = cnpy::npy_load(filename, true);
        int* mmap_data = arr_mmap.data<int>();
        size_t total = arr_mmap.num_vals;
        for (size_t i = 0; i < total; ++i) {
            REQUIRE(mmap_data[i] == static_cast<int>(i * 10));
        }

        // Modify the data in the mmap array
        for (size_t i = 0; i < total; ++i) {
            mmap_data[i] = static_cast<int>(i * 100);
        }
        // arr_mmap goes out of scope, unmapping the file and flushing changes
    }

    // Load the file without mmap and verify the modified values
    cnpy::NpyArray arr_load = cnpy::npy_load(filename, false);
    const int* loaded_data = arr_load.data<int>();
    size_t total = arr_load.num_vals;
    for (size_t i = 0; i < total; ++i) {
        REQUIRE(loaded_data[i] == static_cast<int>(i * 100));
    }

    // Clean up
    std::remove(filename.c_str());
}