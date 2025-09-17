#include "cnpy.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <vector>
#include <string>

TEST_CASE("parse_npy_header recognizes fortran_order True from buffer", "[cnpy]") {
    std::vector<size_t> shape = {2, 3, 4};
    std::vector<char> header = cnpy::create_npy_header<int>(shape);
    std::string s(header.begin(), header.end());
    auto pos = s.find("False");
    REQUIRE(pos != std::string::npos);
    s.replace(pos, 5, "True ");
    std::vector<char> headerTrue(s.begin(), s.end());

    size_t word_size = 0;
    std::vector<size_t> parsed_shape;
    bool fortran_order = false;
    cnpy::parse_npy_header(reinterpret_cast<unsigned char*>(headerTrue.data()),
                           word_size, parsed_shape, fortran_order);

    REQUIRE(fortran_order == true);
    REQUIRE(parsed_shape == shape);
    REQUIRE(word_size == sizeof(int));
}

TEST_CASE("parse_npy_header recognizes fortran_order True from FILE*", "[cnpy]") {
    std::vector<size_t> shape = {3, 1};
    std::vector<char> header = cnpy::create_npy_header<double>(shape);
    std::string s(header.begin(), header.end());
    auto pos = s.find("False");
    REQUIRE(pos != std::string::npos);
    s.replace(pos, 5, "True ");
    std::vector<char> headerTrue(s.begin(), s.end());

    FILE* tmp = std::tmpfile();
    REQUIRE(tmp);
    std::fwrite(headerTrue.data(), 1, headerTrue.size(), tmp);
    std::vector<double> dummy(shape[0] * shape[1]);
    std::fwrite(dummy.data(), sizeof(double), dummy.size(), tmp);
    std::rewind(tmp);

    size_t word_size2 = 0;
    std::vector<size_t> parsed_shape2;
    bool fortran_order2 = false;
    cnpy::parse_npy_header(tmp, word_size2, parsed_shape2, fortran_order2);

    REQUIRE(fortran_order2 == true);
    REQUIRE(parsed_shape2 == shape);
    REQUIRE(word_size2 == sizeof(double));
    std::fclose(tmp);
}

TEST_CASE("npy_load preserves fortran_order flag", "[cnpy]") {
    std::vector<size_t> shape = {2, 2};
    std::vector<char> header = cnpy::create_npy_header<int>(shape);
    std::string s(header.begin(), header.end());
    auto pos = s.find("False");
    REQUIRE(pos != std::string::npos);
    s.replace(pos, 5, "True ");
    std::vector<char> headerTrue(s.begin(), s.end());

    std::string filename = "test_npy_fortran.npy";
    FILE* fp = std::fopen(filename.c_str(), "wb");
    REQUIRE(fp);
    std::fwrite(headerTrue.data(), 1, headerTrue.size(), fp);

    size_t nels = shape[0] * shape[1];
    std::vector<int> data(nels);
    for (size_t i = 0; i < nels; ++i) data[i] = static_cast<int>(i);
    std::fwrite(data.data(), sizeof(int), data.size(), fp);
    std::fclose(fp);

    cnpy::NpyArray arr = cnpy::npy_load(filename);
    REQUIRE(arr.fortran_order == true);
    REQUIRE(arr.shape == shape);
    REQUIRE(arr.as_vec<int>() == data);

    std::remove(filename.c_str());
}