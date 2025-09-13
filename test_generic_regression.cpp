// Generic regression test for cnpy API
#include "cnpy.h"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <typeinfo>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <functional>

TEST_CASE("Generic regression test for cnpy API", "[cnpy][regression]") {
    // Prepare temporary file names
    const std::string npy_file = "temp_test.npy";
    const std::string npz_file = "temp_test.npz";
    const std::string mmap_file = "temp_mmap.npy";

    // Ensure cleanup at the end
    auto cleanup = [&]() {
        std::remove(npy_file.c_str());
        std::remove(npz_file.c_str());
        std::remove(mmap_file.c_str());
    };
    // In case of early exit, use RAII
    struct Cleanup { std::function<void()> f; ~Cleanup(){ f(); } };
    Cleanup guard{cleanup};

    // Individual API calls with INFO and REQUIRE_NOTHROW
    INFO("BigEndianTest");
    REQUIRE_NOTHROW(cnpy::BigEndianTest());

    INFO("map_type_int");
    REQUIRE_NOTHROW(cnpy::map_type(typeid(int)));

    INFO("create_npy_header_int");
    REQUIRE_NOTHROW(cnpy::create_npy_header<int>({1}));

    INFO("npy_save_int");
    REQUIRE_NOTHROW(cnpy::npy_save<int>(npy_file, std::vector<int>{1}));

    INFO("npy_load");
    REQUIRE_NOTHROW(cnpy::npy_load(npy_file));

    INFO("npz_save_int");
    REQUIRE_NOTHROW(cnpy::npz_save<int>(npz_file, "arr", std::vector<int>{1}));

    INFO("npz_load");
    REQUIRE_NOTHROW(cnpy::npz_load(npz_file));

    INFO("npz_load_varname");
    REQUIRE_NOTHROW(cnpy::npz_load(npz_file, "arr"));

    // parse_npy_header from FILE*
    {
        FILE* fp = fopen(npy_file.c_str(), "rb");
        REQUIRE(fp != nullptr);
        size_t ws; std::vector<size_t> sh; bool fo;
        REQUIRE_NOTHROW(cnpy::parse_npy_header(fp, ws, sh, fo));
        fclose(fp);
    }

    // parse_npy_header from buffer
    {
        FILE* fp = fopen(npy_file.c_str(), "rb");
        REQUIRE(fp != nullptr);
        unsigned char buf[256];
        size_t read = fread(buf, 1, sizeof(buf), fp);
        (void)read;
        size_t ws; std::vector<size_t> sh; bool fo;
        REQUIRE_NOTHROW(cnpy::parse_npy_header(buf, ws, sh, fo));
        fclose(fp);
    }

    // parse_zip_footer
    {
        FILE* fp = fopen(npz_file.c_str(), "rb");
        REQUIRE(fp != nullptr);
        uint16_t nrecs; size_t ghs, goff;
        REQUIRE_NOTHROW(cnpy::parse_zip_footer(fp, nrecs, ghs, goff));
        fclose(fp);
    }

    INFO("new_mmap_int");
    REQUIRE_NOTHROW(cnpy::new_mmap<int>(mmap_file, {1}, false));

    // Test operator+= overloads
    using cnpy::operator+=;
    {
        std::vector<char> v;
        REQUIRE_NOTHROW(v += std::string("test"));
    }
    {
        std::vector<char> v;
        REQUIRE_NOTHROW(v += "cstr");
    }
}