// test_mmap_util.cpp
#include "mmap_util.h"
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <utility>

TEST_CASE("MMapFile read-only mapping", "[mmap]") {
    const std::string filename = "test_mmap_file.bin";
    // Create a temporary file with known content
    {
        std::ofstream ofs(filename, std::ios::binary);
        std::string data = "HelloWorld";
        ofs.write(data.c_str(), data.size());
    }

    // Open file in read-only mode
    cnpy::MMapFile mmapFile(filename, "r");
    REQUIRE(mmapFile.is_open());
    REQUIRE(mmapFile.is_readonly());
    REQUIRE(mmapFile.size() == 10);
    REQUIRE(std::memcmp(mmapFile.data(), "HelloWorld", 10) == 0);

    // Move constructor
    cnpy::MMapFile moved(std::move(mmapFile));
    REQUIRE(moved.is_open());
    REQUIRE(!mmapFile.is_open()); // original should be empty

    // Clean up
    std::remove(filename.c_str());
}

TEST_CASE("MMapFile read-write mapping and move assignment", "[mmap]") {
    const std::string filename = "test_mmap_file_rw.bin";
    // Create a temporary file with known content
    {
        std::ofstream ofs(filename, std::ios::binary);
        std::string data = "ABCDEFGHIJ";
        ofs.write(data.c_str(), data.size());
    }

    // Open file in read-write mode
    cnpy::MMapFile mmapFile(filename, "rw");
    REQUIRE(mmapFile.is_open());
    REQUIRE(!mmapFile.is_readonly());
    REQUIRE(mmapFile.size() == 10);

    // Modify the mapped data
    std::memcpy(mmapFile.data(), "12345", 5);
    REQUIRE(std::memcmp(mmapFile.data(), "12345FGHIJ", 10) == 0);

    // Move assignment
    cnpy::MMapFile other(filename, "r");
    other = std::move(mmapFile);
    REQUIRE(other.is_open());
    REQUIRE(!mmapFile.is_open());

    // Ensure data is still correct after move
    REQUIRE(std::memcmp(other.data(), "12345FGHIJ", 10) == 0);

    // other will be destroyed at end of scope, flushing changes
}

// Verify that changes were written to disk
TEST_CASE("MMapFile changes persisted to file", "[mmap]") {
    const std::string filename = "test_mmap_file_rw.bin";
    std::ifstream ifs(filename, std::ios::binary);
    std::string fileContent(10, '\0');
    ifs.read(&fileContent[0], 10);
    REQUIRE(fileContent == "12345FGHIJ");
    std::remove(filename.c_str());
}

TEST_CASE("MMapFile mapping from file descriptor", "[mmap]") {
    const std::string filename = "test_mmap_fd.bin";
    // Create a temporary file with known content
    {
        std::ofstream ofs(filename, std::ios::binary);
        std::string data = "DataFDTest";
        ofs.write(data.c_str(), data.size());
    }
    // Open file descriptor for read-only
    int fd = ::open(filename.c_str(), O_RDONLY);
    REQUIRE(fd >= 0);
    cnpy::MMapFile mmapFd(fd, "r");
    REQUIRE(mmapFd.is_open());
    REQUIRE(mmapFd.is_readonly());
    REQUIRE(mmapFd.size() == 10);
    REQUIRE(std::memcmp(mmapFd.data(), "DataFDTest", 10) == 0);
    // Invalid file descriptors should throw
    REQUIRE_THROWS_AS(cnpy::MMapFile(-1, "r"), std::invalid_argument);
    REQUIRE_THROWS_AS(cnpy::MMapFile(STDIN_FILENO, "r"), std::invalid_argument);
    ::close(fd);
    std::remove(filename.c_str());
}