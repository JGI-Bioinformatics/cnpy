#include "cnpy.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <complex>

TEST_CASE("Placeholder test", "[example]") { REQUIRE(true); }

TEST_CASE("BigEndianTest returns correct endianness indicator", "[cnpy]") {
    int x = 1;
    char expected = ((char*)&x)[0] ? '<' : '>';
    char result = cnpy::BigEndianTest();
    REQUIRE(result == expected);
}
TEST_CASE("map_type returns correct type codes", "[cnpy]") {
    // Floating point types should map to 'f'
    REQUIRE(cnpy::map_type(typeid(float)) == 'f');
    REQUIRE(cnpy::map_type(typeid(double)) == 'f');
    REQUIRE(cnpy::map_type(typeid(long double)) == 'f');

    // Signed integer types should map to 'i'
    REQUIRE(cnpy::map_type(typeid(int)) == 'i');
    REQUIRE(cnpy::map_type(typeid(char)) == 'i');
    REQUIRE(cnpy::map_type(typeid(short)) == 'i');
    REQUIRE(cnpy::map_type(typeid(long)) == 'i');
    REQUIRE(cnpy::map_type(typeid(long long)) == 'i');

    // Unsigned integer types should map to 'u'
    REQUIRE(cnpy::map_type(typeid(unsigned char)) == 'u');
    REQUIRE(cnpy::map_type(typeid(unsigned short)) == 'u');
    REQUIRE(cnpy::map_type(typeid(unsigned int)) == 'u');
    REQUIRE(cnpy::map_type(typeid(unsigned long)) == 'u');
    REQUIRE(cnpy::map_type(typeid(unsigned long long)) == 'u');

    // Boolean type should map to 'b'
    REQUIRE(cnpy::map_type(typeid(bool)) == 'b');

    // Complex types should map to 'c'
    REQUIRE(cnpy::map_type(typeid(std::complex<float>)) == 'c');
    REQUIRE(cnpy::map_type(typeid(std::complex<double>)) == 'c');
    REQUIRE(cnpy::map_type(typeid(std::complex<long double>)) == 'c');

    // Unknown type should map to '?'
    struct Dummy {};
    REQUIRE(cnpy::map_type(typeid(Dummy)) == '?');
}

// Additional unit tests for cnpy::NpyArray
TEST_CASE("NpyArray default constructor", "[cnpy]") {
    cnpy::NpyArray arr;
    REQUIRE(arr.shape.empty());
    REQUIRE(arr.word_size == 0);
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == 0);
    REQUIRE(!arr.data_holder);
}

TEST_CASE("NpyArray constructors and data access", "[cnpy]") {
    std::vector<size_t> shape = {2, 3};
    size_t word_size = sizeof(int);
    bool fortran = false;
    cnpy::NpyArray arr(shape, word_size, fortran);

    // Verify shape and properties
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == word_size);
    REQUIRE(arr.fortran_order == fortran);

    // Compute expected number of elements
    size_t expected_num_vals = 1;
    for (size_t dim : shape) {
        expected_num_vals *= dim;
    }
    REQUIRE(arr.num_vals == expected_num_vals);

    // Verify total byte size
    REQUIRE(arr.num_bytes() == expected_num_vals * word_size);

    // Test mutable data access
    int* data = arr.data<int>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        data[i] = static_cast<int>(i * 10);
    }

    // Verify values via const accessor
    const cnpy::NpyArray& const_arr = arr;
    const int* const_data = const_arr.data<int>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        REQUIRE(const_data[i] == static_cast<int>(i * 10));
    }
}
// Unit tests for cnpy::create_npy_header

TEST_CASE("create_npy_header generates correct header for int type and "
          "multi-dimensional shape",
          "[cnpy]") {
    std::vector<size_t> shape = {2, 3};
    // Generate header for int (4-byte) data
    std::vector<char> header = cnpy::create_npy_header<int>(shape);

    // Basic header checks
    REQUIRE(header.size() >= 10);
    // Magic number
    REQUIRE(static_cast<unsigned char>(header[0]) == 0x93);
    // Magic string "NUMPY"
    REQUIRE(std::string(header.begin() + 1, header.begin() + 6) == "NUMPY");
    // Version 1.0
    REQUIRE(header[6] == 1);
    REQUIRE(header[7] == 0);

    // Extract dict length (little endian)
    uint16_t dict_len =
        static_cast<uint16_t>(static_cast<unsigned char>(header[8]) | (static_cast<unsigned char>(header[9]) << 8));
    REQUIRE(dict_len == header.size() - 10);

    // Extract dict string (including trailing newline)
    std::string dict(header.begin() + 10, header.end());

    // Verify dict content
    // Expected descr: little-endian '<', type code 'i', size 4 -> "<i4"
    REQUIRE(dict.find("'descr': '<i4'") != std::string::npos);
    REQUIRE(dict.find("'fortran_order': False") != std::string::npos);
    REQUIRE(dict.find("'shape': (2, 3)") != std::string::npos);
    // Dict should end with a newline
    REQUIRE(dict.back() == '\n');

    // Header total size must be a multiple of 16 bytes
    REQUIRE((header.size() % 16) == 0);
}

TEST_CASE("create_npy_header generates correct header for double type and "
          "single-dimensional shape",
          "[cnpy]") {
    std::vector<size_t> shape = {5};
    // Generate header for double (8-byte) data
    std::vector<char> header = cnpy::create_npy_header<double>(shape);

    // Basic header checks
    REQUIRE(header.size() >= 10);
    REQUIRE(static_cast<unsigned char>(header[0]) == 0x93);
    REQUIRE(std::string(header.begin() + 1, header.begin() + 6) == "NUMPY");
    REQUIRE(header[6] == 1);
    REQUIRE(header[7] == 0);

    uint16_t dict_len =
        static_cast<uint16_t>(static_cast<unsigned char>(header[8]) | (static_cast<unsigned char>(header[9]) << 8));
    REQUIRE(dict_len == header.size() - 10);

    std::string dict(header.begin() + 10, header.end());

    // Expected descr: little-endian '<', type code 'f', size 8 -> "<f8"
    REQUIRE(dict.find("'descr': '<f8'") != std::string::npos);
    REQUIRE(dict.find("'fortran_order': False") != std::string::npos);
    // For a single dimension, a trailing comma is added after the size
    REQUIRE(dict.find("'shape': (5,)") != std::string::npos);
    REQUIRE(dict.back() == '\n');
    REQUIRE((header.size() % 16) == 0);
}
// Unit tests for cnpy::parse_npy_header (buffer and FILE* overloads)

TEST_CASE("parse_npy_header from buffer extracts correct metadata", "[cnpy]") {
    // Create a header for an int array with shape {2,3}
    std::vector<size_t> shape = {2, 3};
    std::vector<char> header = cnpy::create_npy_header<int>(shape);

    size_t word_size = 0;
    std::vector<size_t> parsed_shape;
    bool fortran_order = true; // initialize to a non‑false value

    // Call the buffer overload
    cnpy::parse_npy_header(reinterpret_cast<unsigned char*>(header.data()), word_size, parsed_shape, fortran_order);

    REQUIRE(word_size == sizeof(int));
    REQUIRE(parsed_shape == shape);
    REQUIRE(fortran_order == false);
}

TEST_CASE("parse_npy_header from FILE* extracts correct metadata", "[cnpy]") {
    // Create a header for a double array with shape {5}
    std::vector<size_t> shape = {5};
    std::vector<char> header = cnpy::create_npy_header<double>(shape);

    // Write the header to a temporary file
    FILE* tmp = std::tmpfile();
    REQUIRE(tmp != nullptr);
    size_t written = std::fwrite(header.data(), 1, header.size(), tmp);
    REQUIRE(written == header.size());

    // Rewind to the beginning of the file before parsing
    std::rewind(tmp);

    size_t word_size = 0;
    std::vector<size_t> parsed_shape;
    bool fortran_order = true;

    // Call the FILE* overload
    cnpy::parse_npy_header(tmp, word_size, parsed_shape, fortran_order);

    REQUIRE(word_size == sizeof(double));
    REQUIRE(parsed_shape == shape);
    REQUIRE(fortran_order == false);

    std::fclose(tmp);
}
// Unit test for cnpy::parse_zip_footer

TEST_CASE("parse_zip_footer correctly parses zip footer", "[cnpy]") {
    // Prepare a simple integer array and shape
    std::vector<int> data = {1, 2, 3, 4};
    std::vector<size_t> shape = {data.size()};
    std::string zip_path = "test_npz.zip";

    // Write the array to a .npz file using the library's npz_save
    cnpy::npz_save<int>(zip_path, "arr", data.data(), shape);

    // Open the generated zip file for reading
    FILE* fp = std::fopen(zip_path.c_str(), "rb");
    REQUIRE(fp != nullptr);

    uint16_t nrecs = 0;
    size_t global_header_size = 0;
    size_t global_header_offset = 0;

    // Parse the zip footer
    cnpy::parse_zip_footer(fp, nrecs, global_header_size, global_header_offset);

    // Verify that exactly one record (array) is present
    REQUIRE(nrecs == 1);
    // The global header size and offset should be non‑zero
    REQUIRE(global_header_size > 0);
    REQUIRE(global_header_offset > 0);

    // Verify that the footer is positioned correctly at the end of the file:
    // file size = offset of global header + size of global header + footer (22
    // bytes)
    std::fseek(fp, 0, SEEK_END);
    long file_size = std::ftell(fp);
    REQUIRE(file_size == static_cast<long>(global_header_offset + global_header_size + 22));

    std::fclose(fp);
    // Clean up the temporary zip file
    std::remove(zip_path.c_str());
}

// Unit test for cnpy::npy_load

// Additional unit tests for cnpy::npy_load with various data types
TEST_CASE("npy_load correctly loads a saved .npy file for int type", "[cnpy]") {
    std::vector<int> data = {2, -3, 4, 6, 7, 0};
    std::string filename = "test_npy_load_int.npy";

    cnpy::npy_save<int>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(int));

    const int* loaded = arr.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }

    std::remove(filename.c_str());
}

// Additional unit tests for cnpy::npy_load with various data types
TEST_CASE("npy_load correctly loads a saved .npy file for unsigned short type", "[cnpy]") {
    std::vector<unsigned short> data = {2, 3, 4, 6, 7, 0};
    std::string filename = "test_npy_load_unsigned_short.npy";

    cnpy::npy_save<unsigned short>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(unsigned short));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(unsigned short));

    const unsigned short* loaded = arr.data<unsigned short>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }

    std::remove(filename.c_str());
}
// Additional unit tests for cnpy::npy_load with various data types
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
    std::string filename = "test_npy_load_longdouble.npy";

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
        REQUIRE(loaded[i] == Catch::Approx(data[i]));
    }

    std::remove(filename.c_str());
}
TEST_CASE("npy_load correctly loads a saved .npy file for complex<float> type", "[cnpy]") {
    std::vector<std::complex<float>> data = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}};
    std::string filename = "test_npy_load_complex_float.npy";

    cnpy::npy_save<std::complex<float>>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(std::complex<float>));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(std::complex<float>));

    const std::complex<float>* loaded = arr.data<std::complex<float>>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i].real() == Catch::Approx(data[i].real()));
        REQUIRE(loaded[i].imag() == Catch::Approx(data[i].imag()));
    }

    std::remove(filename.c_str());
}
TEST_CASE("npy_load correctly loads a saved .npy file for unsigned char type", "[cnpy]") {
    std::vector<unsigned char> data = {0, 255, 128, 64};
    std::string filename = "test_npy_load_uchar.npy";

    cnpy::npy_save<unsigned char>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);

    REQUIRE(arr.shape.size() == 1);
    REQUIRE(arr.shape[0] == data.size());
    REQUIRE(arr.word_size == sizeof(unsigned char));
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.num_vals == data.size());
    REQUIRE(arr.num_bytes() == data.size() * sizeof(unsigned char));

    const unsigned char* loaded = arr.data<unsigned char>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }

    std::remove(filename.c_str());
}
TEST_CASE("npy_load correctly loads a saved .npy file for long type", "[cnpy]") {
    std::vector<long> data = {10, 20, 30, 40, 50};
    std::string filename = "test_npy_load_long.npy";

    // Save the data to a .npy file using the library's npy_save overload for
    // vectors
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
// Unit tests for cnpy::npz_load

TEST_CASE("npz_load with zero arrays throws exception", "[cnpy]") {
    std::string empty_npz = "empty.npz";
    // Create an empty file
    FILE* fp = std::fopen(empty_npz.c_str(), "wb");
    REQUIRE(fp != nullptr);
    std::fclose(fp);
    REQUIRE_THROWS_AS(cnpy::npz_load(empty_npz), std::runtime_error);
    std::remove(empty_npz.c_str());
}

TEST_CASE("npz_load with one array loads correctly", "[cnpy]") {
    std::string filename = "test_npz_one.npz";
    std::vector<int> data = {1, 2, 3, 4};
    std::vector<size_t> shape = {data.size()};
    cnpy::npz_save<int>(filename, "arr1", data.data(), shape);
    cnpy::npz_t arrays = cnpy::npz_load(filename);
    REQUIRE(arrays.size() == 1);
    REQUIRE(arrays.count("arr1") == 1);
    const cnpy::NpyArray& arr = arrays.at("arr1");
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    const int* loaded = arr.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    // Test varname overload
    cnpy::NpyArray arr2 = cnpy::npz_load(filename, "arr1");
    REQUIRE(arr2.shape == shape);
    REQUIRE(arr2.word_size == sizeof(int));
    const int* loaded2 = arr2.data<int>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded2[i] == data[i]);
    }
    std::remove(filename.c_str());
}

TEST_CASE("npz_load with two arrays loads correctly", "[cnpy]") {
    std::string filename = "test_npz_two.npz";
    // First array (int)
    std::vector<int> data_int = {10, 20, 30};
    std::vector<size_t> shape_int = {data_int.size()};
    cnpy::npz_save<int>(filename, "int_arr", data_int.data(), shape_int, "w");
    // Second array (double)
    std::vector<double> data_double = {0.1, 0.2, 0.3, 0.4};
    std::vector<size_t> shape_double = {data_double.size()};
    cnpy::npz_save<double>(filename, "double_arr", data_double.data(), shape_double, "a");
    cnpy::npz_t arrays = cnpy::npz_load(filename);
    REQUIRE(arrays.size() == 2);
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
    std::remove(filename.c_str());
}

TEST_CASE("npz_load with three arrays loads correctly", "[cnpy]") {
    std::string filename = "test_npz_three.npz";
    // First array (int)
    std::vector<int> data_int = {5, 6, 7, 8};
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
// Unit test for npy_load with multi-dimensional data
TEST_CASE("npy_load correctly loads a multi-dimensional .npy file for int type", "[cnpy]") {
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

    // Clean up the generated file
    std::remove(filename.c_str());
}
// Unit test for npz_save with multiple arrays (including a multi-dimensional array)
TEST_CASE("npz_save writes multiple arrays (including multi-dimensional) and npz_load retrieves them correctly",
          "[cnpy]") {
    // Prepare data for a 1‑D integer array
    std::vector<int> int_data = {1, 2, 3, 4, 5};
    // Prepare data for a 2×3 double array (6 elements)
    std::vector<double> double_data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<size_t> double_shape = {2, 3};
    std::string filename = "test_npz_save_multi.npz";

    // Save the integer array using the vector overload (creates a 1‑D .npy inside the .npz)
    cnpy::npz_save<int>(filename, "int_arr", int_data, "w");

    // Append the multi‑dimensional double array using the pointer overload
    cnpy::npz_save<double>(filename, "double_arr", double_data.data(), double_shape, "a");

    // Load all arrays from the .npz file
    cnpy::npz_t arrays = cnpy::npz_load(filename);
    REQUIRE(arrays.size() == 2);
    REQUIRE(arrays.count("int_arr") == 1);
    REQUIRE(arrays.count("double_arr") == 1);

    // Verify the integer array
    const cnpy::NpyArray& int_arr = arrays.at("int_arr");
    REQUIRE(int_arr.shape == std::vector<size_t>{int_data.size()});
    REQUIRE(int_arr.word_size == sizeof(int));
    REQUIRE(int_arr.fortran_order == false);
    const int* loaded_int = int_arr.data<int>();
    for (size_t i = 0; i < int_data.size(); ++i) {
        REQUIRE(loaded_int[i] == int_data[i]);
    }

    // Verify the double array (multi‑dimensional)
    const cnpy::NpyArray& double_arr = arrays.at("double_arr");
    REQUIRE(double_arr.shape == double_shape);
    REQUIRE(double_arr.word_size == sizeof(double));
    REQUIRE(double_arr.fortran_order == false);
    const double* loaded_double = double_arr.data<double>();
    for (size_t i = 0; i < double_data.size(); ++i) {
        REQUIRE(loaded_double[i] == Catch::Approx(double_data[i]));
    }

    // Verify loading a single array via the varname overload
    cnpy::NpyArray int_arr_single = cnpy::npz_load(filename, "int_arr");
    REQUIRE(int_arr_single.shape == std::vector<size_t>{int_data.size()});
    const int* loaded_int_single = int_arr_single.data<int>();
    for (size_t i = 0; i < int_data.size(); ++i) {
        REQUIRE(loaded_int_single[i] == int_data[i]);
    }

    // Clean up the temporary .npz file
    std::remove(filename.c_str());
}
// Unit test for npy_save append using pointer overload (multi-dimensional)
TEST_CASE("npy_save append works with pointer overload for multi-dimensional data", "[cnpy]") {
    // Initial 2x2 array
    std::vector<int> data1 = {0, 1, 2, 3};
    std::vector<size_t> shape1 = {2, 2};
    std::string filename = "test_npy_save_append_ptr.npy";

    // Save initial data (write mode)
    cnpy::npy_save<int>(filename, data1.data(), shape1, "w");

    // Append additional 1x2 data
    std::vector<int> data2 = {4, 5};
    std::vector<size_t> shape2 = {1, 2};
    cnpy::npy_save<int>(filename, data2.data(), shape2, "a");

    // Load and verify combined shape and data
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    std::vector<size_t> expected_shape = {3, 2};
    REQUIRE(arr.shape == expected_shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    const int* loaded = arr.data<int>();
    std::vector<int> expected = {0, 1, 2, 3, 4, 5};
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(loaded[i] == expected[i]);
    }

    // Clean up
    std::remove(filename.c_str());
}

// Unit test for npy_save append using vector overload (1‑D)
TEST_CASE("npy_save append works with vector overload for 1‑D data", "[cnpy]") {
    // Initial data
    std::vector<int> data1 = {1, 2, 3};
    std::string filename = "test_npy_save_append_vec.npy";

    // Save initial data (write mode)
    cnpy::npy_save<int>(filename, data1, "w");

    // Append additional data
    std::vector<int> data2 = {4, 5, 6, 7};
    cnpy::npy_save<int>(filename, data2, "a");

    // Load and verify combined shape and data
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    std::vector<size_t> expected_shape = {7};
    REQUIRE(arr.shape == expected_shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == false);
    const int* loaded = arr.data<int>();
    std::vector<int> expected = {1, 2, 3, 4, 5, 6, 7};
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(loaded[i] == expected[i]);
    }

    // Clean up
    std::remove(filename.c_str());
}

// Unit test for mmap-backed NpyArray constructor
TEST_CASE("NpyArray mmap constructor", "[cnpy]") {
    std::vector<size_t> shape = {2, 3};
    bool fortran = false;
    std::string filename = "test_npy_mmap.npy";

    // Create a new mmap-backed NpyArray using the library helper
    cnpy::NpyArray arr = cnpy::new_mmap<int>(filename, shape, fortran);

    // Verify shape and properties
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == fortran);

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
TEST_CASE("new_mmap unsigned short 1D", "[cnpy]") {
    std::vector<size_t> shape = {10};
    std::string filename = "test_npy_mmap_ushort.npy";

    cnpy::NpyArray arr = cnpy::new_mmap<unsigned short>(filename, shape, false);

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

TEST_CASE("new_mmap int 2D", "[cnpy]") {
    std::vector<size_t> shape = {4, 5};
    std::string filename = "test_npy_mmap_int_2d.npy";

    cnpy::NpyArray arr = cnpy::new_mmap<int>(filename, shape, false);

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
        data[i] = static_cast<int>(i * 5);
    }

    // Verify values via const accessor
    const cnpy::NpyArray& const_arr = arr;
    const int* const_data = const_arr.data<int>();
    for (size_t i = 0; i < expected_num_vals; ++i) {
        REQUIRE(const_data[i] == static_cast<int>(i * 5));
    }

    // Clean up the temporary file
    std::remove(filename.c_str());
}

TEST_CASE("new_mmap double 3D", "[cnpy]") {
    std::vector<size_t> shape = {2, 3, 4};
    std::string filename = "test_npy_mmap_double_3d.npy";

    cnpy::NpyArray arr = cnpy::new_mmap<double>(filename, shape, false);

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
// Test for npz_save with compression option
TEST_CASE("npz_save with compression option compresses data correctly", "[cnpy]") {
    std::vector<int> data = {10, 20, 30, 40, 50, 60};
    std::vector<size_t> shape = {2, 3};
    std::string filename = "test_npz_compress.npz";

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
// New test cases for additional npy/npz types

TEST_CASE("npy_save/load for char type", "[cnpy]") {
    std::vector<char> data = {'a', 'b', '\0', 'z'};
    std::string filename = "test_npy_char.npy";
    cnpy::npy_save<char>(filename, data);
    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.word_size == sizeof(char));
    const char* loaded = arr.data<char>();
    for (size_t i = 0; i < data.size(); ++i) {
        REQUIRE(loaded[i] == data[i]);
    }
    std::remove(filename.c_str());
}

// Tests cnpy::npy_save/load for unsigned int and unsigned long long types
TEST_CASE("npy_save/load for unsigned int and unsigned long long types", "[cnpy]") {
    {
        std::vector<unsigned int> data = {0u, 1u, std::numeric_limits<unsigned int>::max()};
        std::string filename = "test_npy_uint.npy";
        cnpy::npy_save<unsigned int>(filename, data);
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        REQUIRE(arr.word_size == sizeof(unsigned int));
        const unsigned int* loaded = arr.data<unsigned int>();
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE(loaded[i] == data[i]);
        }
        std::remove(filename.c_str());
    }
    {
        std::vector<unsigned long long> data = {0ULL, 1ULL, std::numeric_limits<unsigned long long>::max()};
        std::string filename = "test_npy_ull.npy";
        cnpy::npy_save<unsigned long long>(filename, data);
        cnpy::NpyArray arr = cnpy::npy_load(filename);
        REQUIRE(arr.word_size == sizeof(unsigned long long));
        const unsigned long long* loaded = arr.data<unsigned long long>();
        for (size_t i = 0; i < data.size(); ++i) {
            REQUIRE(loaded[i] == data[i]);
        }
        std::remove(filename.c_str());
    }
}
