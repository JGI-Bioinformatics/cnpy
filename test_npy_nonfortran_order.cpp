#include "cnpy.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <cstdio>
#include <string>

TEST_CASE("npy_save and npy_load maintain C-order (non-fortran)", "[cnpy][npy][c-order]") {
    // Prepare a 2×3 array in row-major order
    std::vector<size_t> shape = {2, 3};
    size_t nels = shape[0] * shape[1];
    std::vector<int> data(nels);
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            data[i * shape[1] + j] = static_cast<int>(i * shape[1] + j + 1);
        }
    }

    std::string filename = "test_nonfortran.npy";
    cnpy::npy_save<int>(filename, data.data(), shape);

    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.fortran_order == false);
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.as_vec<int>() == data);

    std::remove(filename.c_str());
}

TEST_CASE("parse_npy_header default fortran_order is False from buffer", "[cnpy][npy][header]") {
    std::vector<size_t> shape = {1, 2, 3};
    std::vector<char> header = cnpy::create_npy_header<float>(shape);

    size_t word_size = 0;
    std::vector<size_t> parsed_shape;
    bool fortran_order = true;
    cnpy::parse_npy_header(reinterpret_cast<unsigned char*>(header.data()),
                           word_size, parsed_shape, fortran_order);

    REQUIRE(fortran_order == false);
    REQUIRE(parsed_shape == shape);
    REQUIRE(word_size == sizeof(float));
}

TEST_CASE("parse_npy_header default fortran_order is False from FILE*", "[cnpy][npy][header]") {
    std::vector<size_t> shape = {4, 1};
    std::vector<char> header = cnpy::create_npy_header<long>(shape);

    FILE* tmp = std::tmpfile();
    REQUIRE(tmp);
    std::fwrite(header.data(), 1, header.size(), tmp);
    std::rewind(tmp);

    size_t word_size2 = 0;
    std::vector<size_t> parsed_shape2;
    bool fortran_order2 = true;
    cnpy::parse_npy_header(tmp, word_size2, parsed_shape2, fortran_order2);

    REQUIRE(fortran_order2 == false);
    REQUIRE(parsed_shape2 == shape);
    REQUIRE(word_size2 == sizeof(long));
    std::fclose(tmp);
}