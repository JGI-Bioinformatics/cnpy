// test_mmap_save_load.cpp
#include "cnpy.h"
#include "mmap_util.h"
#include <catch2/catch_approx.hpp>
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

// Additional tests for cnpy::new_npy_mmap with different data types and shapes
TEST_CASE("new_npy_mmap int 2D", "[cnpy]") {
    std::vector<size_t> shape = {3, 4};
    std::string filename = "test_npy_mmap_int_2d.npy";
    cnpy::NpyArray arr = cnpy::new_npy_mmap<int>(filename, shape, false);
    // Verify shape and properties
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    // Compute expected number of elements
    size_t expected_num_vals = 1;
    for (size_t dim : shape) {
        expected_num_vals *= dim;
    }
    REQUIRE(arr.num_vals == expected_num_vals);
    // Test mutable data access
    int* data = arr.data<int>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        data[i] = static_cast<int>(i * 7);
    }
    // Verify values via const accessor
    const cnpy::NpyArray& const_arr = arr;
    const int* const_data = const_arr.data<int>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        REQUIRE(const_data[i] == static_cast<int>(i * 7));
    }
    // Clean up the temporary file
    std::remove(filename.c_str());
}

TEST_CASE("new_npy_mmap unsigned short 1D", "[cnpy]") {
    std::vector<size_t> shape = {10};
    std::string filename = "test_npy_mmap_ushort.npy";
    cnpy::NpyArray arr = cnpy::new_npy_mmap<unsigned short>(filename, shape, false);
    // Verify shape and properties
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(unsigned short));
    REQUIRE(arr.fortran_order == false);
    // Compute expected number of elements
    size_t expected_num_vals = 1;
    for (size_t dim : shape) {
        expected_num_vals *= dim;
    }
    REQUIRE(arr.num_vals == expected_num_vals);
    // Test mutable data access
    unsigned short* data = arr.data<unsigned short>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        data[i] = static_cast<unsigned short>(i * 3);
    }
    // Verify values via const accessor
    const cnpy::NpyArray& const_arr = arr;
    const unsigned short* const_data = const_arr.data<unsigned short>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        REQUIRE(const_data[i] == static_cast<unsigned short>(i * 3));
    }
    // Clean up the temporary file
    std::remove(filename.c_str());
}

TEST_CASE("new_npy_mmap double 3D", "[cnpy]") {
    std::vector<size_t> shape = {2, 3, 4};
    std::string filename = "test_npy_mmap_double_3d.npy";
    cnpy::NpyArray arr = cnpy::new_npy_mmap<double>(filename, shape, false);
    // Verify shape and properties
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(double));
    REQUIRE(arr.fortran_order == false);
    // Compute expected number of elements
    size_t expected_num_vals = 1;
    for (size_t dim : shape) {
        expected_num_vals *= dim;
    }
    REQUIRE(arr.num_vals == expected_num_vals);
    // Test mutable data access
    double* data = arr.data<double>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        data[i] = static_cast<double>(i) * 0.1;
    }
    // Verify values via const accessor
    const cnpy::NpyArray& const_arr = arr;
    const double* const_data = const_arr.data<double>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        REQUIRE(const_data[i] == static_cast<double>(i) * 0.1);
    }
    // Clean up the temporary file
    std::remove(filename.c_str());
}

// Test for npz_save and npz_load with multiple data types
TEST_CASE("npz_save and npz_load with multiple data types", "[cnpy][npz]") {
    std::string filename = "test_multiple_types.npz";
    // First array (int)
    std::vector<int> data_int = {1, 2, 3, 4};
    std::vector<size_t> shape_int = {data_int.size()};
    cnpy::npz_save<int>(filename, "int_arr", data_int.data(), shape_int, "w");
    // Second array (double)
    std::vector<double> data_double = {1.1, 2.2, 3.3};
    std::vector<size_t> shape_double = {data_double.size()};
    cnpy::npz_save<double>(filename, "double_arr", data_double.data(), shape_double, "a");
    // Third array (float)
    std::vector<float> data_float = {0.5f, 1.5f};
    std::vector<size_t> shape_float = {data_float.size()};
    cnpy::npz_save<float>(filename, "float_arr", data_float.data(), shape_float, "a");
    cnpy::npz_t arrays = cnpy::npz_load(filename);
    REQUIRE(arrays.size() == 3);
    // Verify int_arr
    const cnpy::NpyArray& arr_int = arrays.at("int_arr");
    REQUIRE(arr_int.shape == shape_int);
    REQUIRE(arr_int.word_size == sizeof(int));
    const int* loaded_int = arr_int.data<int>();
    for (size_t i = 0; i < data_int.size(); ++i) {
        REQUIRE(loaded_int[i] == data_int[i]);
    }
    // Verify double_arr
    const cnpy::NpyArray& arr_double = arrays.at("double_arr");
    REQUIRE(arr_double.shape == shape_double);
    REQUIRE(arr_double.word_size == sizeof(double));
    const double* loaded_double = arr_double.data<double>();
    for (size_t i = 0; i < data_double.size(); ++i) {
        REQUIRE(loaded_double[i] == Catch::Approx(data_double[i]));
    }
    // Verify float_arr
    const cnpy::NpyArray& arr_float = arrays.at("float_arr");
    REQUIRE(arr_float.shape == shape_float);
    REQUIRE(arr_float.word_size == sizeof(float));
    const float* loaded_float = arr_float.data<float>();
    for (size_t i = 0; i < data_float.size(); ++i) {
        REQUIRE(loaded_float[i] == Catch::Approx(data_float[i]));
    }
    std::remove(filename.c_str());
}

// Test for npz_save with compression option
TEST_CASE("npz_save with compression option compresses data correctly", "[cnpy]") {
    std::vector<int> data = {10, 20, 30, 40, 50, 60};
    std::vector<size_t> shape = {2, 3};
    std::string filename = "test_npz_compress.npz"; // Changed filename to avoid conflict
    // Save with compression enabled
    cnpy::npz_save<int>(filename, "arr_compressed", data.data(), shape, "w", true);
    // Load and verify
    cnpy::npz_t arrays = cnpy::npz_load(filename);
    REQUIRE(arrays.size() == 1);
    const cnpy::NpyArray& arr = arrays.at("arr_compressed");
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(int));
    const int* loaded = arr.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    // Clean up
    std::remove(filename.c_str());
}

// Unit test for npy_save (int vector)
TEST_CASE("npy_save correctly writes a .npy file for int type and loads correctly", "[cnpy]") {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::string filename = "test_npy_save_int.npy";
    // Save using npy_save
    cnpy::npy_save<int>(filename, data);
    // Load using npy_load
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Verify shape and metadata
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(int));
    // Verify data
    const int* loaded = arr.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    // Clean up
    std::remove(filename.c_str());
}

// Unit test for npy_save with multi-dimensional data
TEST_CASE("npy_save correctly writes a multi-dimensional .npy file and loads correctly", "[cnpy]") {
    std::vector<int> data = {0, 1, 2, 3, 4, 5};
    std::vector<size_t> shape = {2, 3};
    std::string filename = "test_npy_save_multi.npy";
    // Save using pointer overload
    cnpy::npy_save<int>(filename, data.data(), shape);
    // Load using npy_load
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    // Verify shape and metadata
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(int));
    // Verify data
    const int* loaded = arr.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    // Clean up
    std::remove(filename.c_str());
}

// Unit test for npy_load with float, double, and long double types
TEST_CASE("npy_load correctly loads a saved .npy file for float type", "[cnpy]") {
    std::vector<float> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    std::string filename = "test_npy_load_float.npy";
    cnpy::npy_save<float>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(float));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(float));
    const float* loaded = arr.data<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == Catch::Approx(data[i]));
    }
    std::remove(filename.c_str());
}

TEST_CASE("npy_load correctly loads a saved .npy file for double type", "[cnpy]") {
    std::vector<double> data = {0.1, 0.2, 0.3, 0.4, 0.5};
    std::string filename = "test_npy_load_double.npy";
    cnpy::npy_save<double>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(double));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(double));
    const double* loaded = arr.data<double>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == Catch::Approx(data[i]));
    }
    std::remove(filename.c_str());
}

TEST_CASE("npy_load correctly loads a saved .npy file for long double type", "[cnpy]") {
    std::vector<long double> data = {0.1L, 0.2L, 0.3L, 0.4L, 0.5L};
    std::string filename = "test_npy_load_long_double.npy";
    cnpy::npy_save<long double>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(long double));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(long double));
    const long double* loaded = arr.data<long double>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == Catch::Approx(static_cast<double>(data[i])));
    }
    std::remove(filename.c_str());
}

// Tests cnpy::npy_save/load for char type
TEST_CASE("npy_save/load for char type", "[cnpy]") {
    std::vector<char> data = {'a', 'b', '\0', 'z'};
    std::string filename = "test_npy_char.npy";
    cnpy::npy_save<char>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(char));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(char));
    const char* loaded = arr.data<char>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    std::remove(filename.c_str());
}

// Tests cnpy::npy_save/load for unicode type
TEST_CASE("npy_save/load for unicode type", "[cnpy]") {
    std::u32string data = U"Hello, 世界"; // Unicode string with non-ASCII characters
    std::vector<char32_t> vec_data(data.begin(), data.end());
    std::string filename = "test_npy_unicode.npy";
    cnpy::npy_save<char32_t>(filename, vec_data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == vec_data.size());
    REQUIRE(arr.word_size == sizeof(char32_t));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == vec_data.size());
    REQUIRE(arr.num_bytes() == vec_data.size() * sizeof(char32_t));
    const char32_t* loaded = arr.data<char32_t>();
    for (size_t i = 0; i < vec_data.size(); ++i) {
        REQUIRE(loaded[i] == vec_data[i]);
    }
    std::remove(filename.c_str());
}

// Tests cnpy::npy_save/load for unsigned int and unsigned long long types
TEST_CASE("npy_save/load for unsigned int and unsigned long long types", "[cnpy]") {
    {
        std::vector<unsigned int> data = {0u, 1u, std::numeric_limits<unsigned int>::max()};
        std::string filename = "test_npy_unsigned_int.npy";
        cnpy::npy_save<unsigned int>(filename, data);
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        REQUIRE(arr.shape.size() == 1);
        REQUIRE(arr.shape[0] == data.size());
        REQUIRE(arr.word_size == sizeof(unsigned int));
        REQUIRE(arr.fortran_order == false);
        REQUIRE(arr.num_vals == data.size());
        REQUIRE(arr.num_bytes() == data.size() * sizeof(unsigned int));
        const unsigned int* loaded = arr.data<unsigned int>();
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE(loaded[i] == data[i]);
        }
        std::remove(filename.c_str());
    }
    {
        std::vector<unsigned long long> data = {0ull, 1ull, std::numeric_limits<unsigned long long>::max()};
        std::string filename = "test_npy_unsigned_long_long.npy";
        cnpy::npy_save<unsigned long long>(filename, data);
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        REQUIRE(arr.shape.size() == 1);
        REQUIRE(arr.shape[0] == data.size());
        REQUIRE(arr.word_size == sizeof(unsigned long long));
        REQUIRE(arr.fortran_order == false);
        REQUIRE(arr.num_vals == data.size());
        REQUIRE(arr.num_bytes() == data.size() * sizeof(unsigned long long));
        const unsigned long long* loaded = arr.data<unsigned long long>();
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE(loaded[i] == data[i]);
        }
        std::remove(filename.c_str());
    }
}