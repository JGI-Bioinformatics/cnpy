// test_mmap_save_load.cpp
#include "cnpy.h"
#include "mmap_util.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstdlib>
#include <sstream>
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

// Tests for npz_load varname overload with uncompressed .npz files
TEST_CASE("npz_load varname overload uncompressed .npz memory and mmap", "[cnpy][npz]") {
    const std::string filename = "test_uncompressed.npz";
    const std::string varname = "arr";
    std::vector<size_t> shape = {2, 3};
    std::vector<int> data(shape[0] * shape[1]);
    for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<int>(i);

    // Save uncompressed NPZ
    cnpy::npz_save<int>(filename, varname, data.data(), shape);

    // Memory load
    cnpy::NpyArray arr_mem = cnpy::npz_load(filename, varname, false);
    REQUIRE(arr_mem.shape == shape);
    REQUIRE(arr_mem.as_vec<int>() == data);

    // MMap load
    cnpy::NpyArray arr_mmap = cnpy::npz_load(filename, varname, true);
    REQUIRE(arr_mmap.shape == shape);
    REQUIRE(arr_mmap.as_vec<int>() == data);

    // Missing variable throws
    REQUIRE_THROWS_AS(cnpy::npz_load(filename, "missing", false), std::runtime_error);
    REQUIRE_THROWS_AS(cnpy::npz_load(filename, "missing", true), std::runtime_error);

    std::remove(filename.c_str());
}

// Tests for npz_load on NumPy compressed .npz files
TEST_CASE("npz_load from NumPy compressed .npz memory load and error on mmap", "[cnpy][npz][compressed]") {
    const std::string filename = "test_compressed.npz";
    // Generate compressed NPZ via C++ API
    std::vector<size_t> shape = {2, 3};
    std::vector<int> data = {0, 1, 2, 3, 4, 5};
    cnpy::npz_save<int>(filename, "arr", data.data(), shape, "w", true);

    // Memory load should work
    cnpy::NpyArray arr_mem = cnpy::npz_load(filename, "arr", false);
    REQUIRE(arr_mem.shape == shape);
    REQUIRE(arr_mem.as_vec<int>() == data);

    // MMap load not supported for compressed NPZ
    {
        std::ostringstream err;
        auto old_buf = std::cerr.rdbuf(err.rdbuf());
        cnpy::NpyArray arr_mmap = cnpy::npz_load(filename, "arr", true);
        std::cerr.rdbuf(old_buf);
        REQUIRE(err.str().find("Warning: npz_load: memory map requested but file") != std::string::npos);
        REQUIRE(arr_mmap.shape == shape);
        REQUIRE(arr_mmap.as_vec<int>() == data);
    }

    // Missing variable throws in both modes
    REQUIRE_THROWS_AS(cnpy::npz_load(filename, "missing", false), std::runtime_error);
    REQUIRE_THROWS_AS(cnpy::npz_load(filename, "missing", true), std::runtime_error);

    std::remove(filename.c_str());
}
// Test default npz_load overload in mmap mode for uncompressed .npz
TEST_CASE("npz_load default overload mmap loads uncompressed .npz correctly", "[cnpy][npz][mmap]") {
    const std::string filename = "test_npz_default_uncompressed.npz";
    std::vector<int> data1 = {1, 2, 3};
    std::vector<size_t> shape1 = {data1.size()};
    cnpy::npz_save<int>(filename, "arr1", data1.data(), shape1, "w");
    std::vector<int> data2 = {4, 5, 6, 7};
    std::vector<size_t> shape2 = {data2.size()};
    cnpy::npz_save<int>(filename, "arr2", data2.data(), shape2, "a");
    cnpy::npz_t arrays = cnpy::npz_load(filename, true);
    REQUIRE(arrays.size() == 2);
    REQUIRE(arrays.count("arr1") == 1);
    REQUIRE(arrays.count("arr2") == 1);
    REQUIRE(arrays.at("arr1").as_vec<int>() == data1);
    REQUIRE(arrays.at("arr2").as_vec<int>() == data2);
    std::remove(filename.c_str());
}

// Test default npz_load overload in mmap mode fallback on compressed .npz
TEST_CASE("npz_load default overload mmap fallback on compressed .npz", "[cnpy][npz][mmap][compressed]") {
    const std::string filename = "test_npz_default_compressed.npz";
    std::vector<size_t> shape = {3};
    std::vector<int> data = {7, 8, 9};
    cnpy::npz_save<int>(filename, "arr", data.data(), shape, "w", true);
    std::ostringstream err;
    auto old_buf = std::cerr.rdbuf(err.rdbuf());
    cnpy::npz_t arrays = cnpy::npz_load(filename, true);
    std::cerr.rdbuf(old_buf);
    REQUIRE(arrays.size() == 1);
    REQUIRE(arrays.count("arr") == 1);
    REQUIRE(arrays.at("arr").as_vec<int>() == data);
    REQUIRE(err.str().find("Warning: npz_load: memory map requested but file") != std::string::npos);
    std::remove(filename.c_str());
}