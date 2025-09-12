 // test_mmap_util.cpp
 #include "mmap_util.h"
 #include <catch2/catch_test_macros.hpp>
 #include <fstream>
 #include <cstdio>
 #include <cstring>
 #include <utility>
 #include <string>
 
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