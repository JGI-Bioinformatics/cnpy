// test_new_npz_mmap.cpp
#include "cnpy.h"
#include <catch2/catch_test_macros.hpp>
#include <vector>
#include <string>
#include <cstdio>
#include <typeinfo>

TEST_CASE("new_npz_mmap zero arrays", "[cnpy][npz][mmap]") {
    std::string filename = "test_zero.npz";
    std::vector<cnpy::ShapeAndType> shapes;
    auto arrays = cnpy::new_npz_mmap(filename, shapes, false);
    REQUIRE(arrays.empty());
    std::remove(filename.c_str());
}

TEST_CASE("new_npz_mmap single array", "[cnpy][npz][mmap]") {
    std::string filename = "test_single.npz";
    std::vector<cnpy::ShapeAndType> shapes = {{ {2, 3}, typeid(int), "arr" }};
    auto arrays = cnpy::new_npz_mmap(filename, shapes, true);
    REQUIRE(arrays.size() == 1);
    REQUIRE(arrays.count("arr") == 1);
    auto &arr = arrays.at("arr");
    REQUIRE(arr.shape == std::vector<size_t>{2, 3});
    REQUIRE(arr.word_size == sizeof(int));
    REQUIRE(arr.fortran_order == true);
    std::vector<int> data = arr.as_vec<int>();
    REQUIRE(data.size() == 6);
    for (auto v : data) REQUIRE(v == 0);
    std::remove(filename.c_str());
}

TEST_CASE("new_npz_mmap multiple arrays various types and sizes", "[cnpy][npz][mmap]") {
    std::string filename = "test_multi.npz";
    std::vector<cnpy::ShapeAndType> shapes = {
        {{4}, typeid(int), "int_arr"},
        {{2, 2}, typeid(double), "double_arr"},
        {{3}, typeid(char), "char_arr"}
    };
    auto arrays = cnpy::new_npz_mmap(filename, shapes, false);
    REQUIRE(arrays.size() == 3);
    {
        auto &arr = arrays.at("int_arr");
        REQUIRE(arr.shape == std::vector<size_t>{4});
        REQUIRE(arr.word_size == sizeof(int));
        auto vec = arr.as_vec<int>();
        REQUIRE(vec.size() == 4);
        for (auto v : vec) REQUIRE(v == 0);
    }
    {
        auto &arr = arrays.at("double_arr");
        REQUIRE(arr.shape == std::vector<size_t>{2, 2});
        REQUIRE(arr.word_size == sizeof(double));
        auto vec = arr.as_vec<double>();
        REQUIRE(vec.size() == 4);
        for (auto v : vec) REQUIRE(v == 0.0);
    }
    {
        auto &arr = arrays.at("char_arr");
        REQUIRE(arr.shape == std::vector<size_t>{3});
        REQUIRE(arr.word_size == sizeof(char));
        auto vec = arr.as_vec<char>();
        REQUIRE(vec.size() == 3);
        for (auto v : vec) REQUIRE(v == 0);
    }
    std::remove(filename.c_str());
}