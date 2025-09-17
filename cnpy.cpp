// Copyright (C) 2011  Carl Rogers
// Released under MIT License
// license available in LICENSE file, or at
// http://www.opensource.org/licenses/mit-license.php

#include "cnpy.h"
#include "mmap_util.h"
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <stdint.h>

char cnpy::BigEndianTest() {
    int x = 1;
    return (((char*)&x)[0]) ? '<' : '>';
}

char cnpy::map_type(const std::type_info& t) {
    if (t == typeid(float)) return 'f';
    if (t == typeid(double)) return 'f';
    if (t == typeid(long double)) return 'f';

    if (t == typeid(int)) return 'i';
    if (t == typeid(char)) return 'i';
    if (t == typeid(short)) return 'i';
    if (t == typeid(long)) return 'i';
    if (t == typeid(long long)) return 'i';

    if (t == typeid(unsigned char)) return 'u';
    if (t == typeid(unsigned short)) return 'u';
    if (t == typeid(unsigned long)) return 'u';
    if (t == typeid(unsigned long long)) return 'u';
    if (t == typeid(unsigned int)) return 'u';

    if (t == typeid(bool)) return 'b';

    if (t == typeid(std::complex<float>)) return 'c';
    if (t == typeid(std::complex<double>)) return 'c';
    if (t == typeid(std::complex<long double>))
        return 'c';

    else
        return '?';
}

template <> std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const std::string rhs) {
    lhs.insert(lhs.end(), rhs.begin(), rhs.end());
    return lhs;
}

template <> std::vector<char>& cnpy::operator+=(std::vector<char>& lhs, const char* rhs) {
    // write in little endian
    size_t len = strlen(rhs);
    lhs.reserve(len);
    for (size_t byte = 0; byte < len; byte++) {
        lhs.push_back(rhs[byte]);
    }
    return lhs;
}

void cnpy::parse_npy_header(unsigned char* buffer, size_t& word_size, Shape& shape, bool& fortran_order) {
    std::string magic_string((const char*)buffer, 6);
    uint8_t major_version = *reinterpret_cast<uint8_t*>(buffer + 6);
    uint8_t minor_version = *reinterpret_cast<uint8_t*>(buffer + 7);
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer + 8);
    std::string header(reinterpret_cast<char*>(buffer + 9), header_len);

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order") + 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr") + 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

void cnpy::parse_npy_header(FILE* fp, size_t& word_size, Shape& shape, bool& fortran_order) {
    char buffer[256];
    size_t res = fread(buffer, sizeof(char), 11, fp);
    if (res != 11) throw std::runtime_error("parse_npy_header: failed fread");
    std::string header = fgets(buffer, 256, fp);
    assert(header[header.size() - 1] == '\n');

    size_t loc1, loc2;

    // fortran order
    loc1 = header.find("fortran_order");
    if (loc1 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: 'fortran_order'");
    loc1 += 16;
    fortran_order = (header.substr(loc1, 4) == "True" ? true : false);

    // shape
    loc1 = header.find("(");
    loc2 = header.find(")");
    if (loc1 == std::string::npos || loc2 == std::string::npos)
        throw std::runtime_error("parse_npy_header: failed to find header keyword: '(' or ')'");

    std::regex num_regex("[0-9][0-9]*");
    std::smatch sm;
    shape.clear();

    std::string str_shape = header.substr(loc1 + 1, loc2 - loc1 - 1);
    while (std::regex_search(str_shape, sm, num_regex)) {
        shape.push_back(std::stoi(sm[0].str()));
        str_shape = sm.suffix().str();
    }

    // endian, word size, data type
    // byte order code | stands for not applicable.
    // not sure when this applies except for byte array
    loc1 = header.find("descr");
    if (loc1 == std::string::npos) throw std::runtime_error("parse_npy_header: failed to find header keyword: 'descr'");
    loc1 += 9;
    bool littleEndian = (header[loc1] == '<' || header[loc1] == '|' ? true : false);
    assert(littleEndian);

    // char type = header[loc1+1];
    // assert(type == map_type(T));

    std::string str_ws = header.substr(loc1 + 2);
    loc2 = str_ws.find("'");
    word_size = atoi(str_ws.substr(0, loc2).c_str());
}

void cnpy::parse_zip_footer(FILE* fp, uint16_t& nrecs, size_t& global_header_size, size_t& global_header_offset) {
    std::vector<char> footer(22);
    fseek(fp, -22, SEEK_END);
    size_t res = fread(&footer[0], sizeof(char), 22, fp);
    if (res != 22) throw std::runtime_error("parse_zip_footer: failed fread");

    uint16_t disk_no, disk_start, nrecs_on_disk, comment_len;
    disk_no = *(uint16_t*)&footer[4];
    disk_start = *(uint16_t*)&footer[6];
    nrecs_on_disk = *(uint16_t*)&footer[8];
    nrecs = *(uint16_t*)&footer[10];
    global_header_size = *(uint32_t*)&footer[12];
    global_header_offset = *(uint32_t*)&footer[16];
    comment_len = *(uint16_t*)&footer[20];

    assert(disk_no == 0);
    assert(disk_start == 0);
    assert(nrecs_on_disk == nrecs);
    assert(comment_len == 0);
}

cnpy::NpyArray load_the_npy_file(FILE* fp) {
    cnpy::Shape shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(fp, word_size, shape, fortran_order);

    cnpy::NpyArray arr(shape, word_size, fortran_order);
    size_t nread = fread(arr.data<char>(), 1, arr.num_bytes(), fp);
    if (nread != arr.num_bytes()) throw std::runtime_error("load_the_npy_file: failed fread");
    return arr;
}

cnpy::NpyArray load_the_npz_array(FILE* fp, uint32_t compr_bytes, uint32_t uncompr_bytes) {

    std::vector<unsigned char> buffer_compr(compr_bytes);
    std::vector<unsigned char> buffer_uncompr(uncompr_bytes);
    size_t nread = fread(&buffer_compr[0], 1, compr_bytes, fp);
    if (nread != compr_bytes) throw std::runtime_error("load_the_npy_file: failed fread");

    int err;
    z_stream d_stream;

    d_stream.zalloc = Z_NULL;
    d_stream.zfree = Z_NULL;
    d_stream.opaque = Z_NULL;
    d_stream.avail_in = 0;
    d_stream.next_in = Z_NULL;
    err = inflateInit2(&d_stream, -MAX_WBITS);
    if (err != Z_OK) throw std::runtime_error("load_the_npz_array: inflateInit2 failed");

    d_stream.avail_in = compr_bytes;
    d_stream.next_in = &buffer_compr[0];
    d_stream.avail_out = uncompr_bytes;
    d_stream.next_out = &buffer_uncompr[0];

    err = inflate(&d_stream, Z_FINISH);
    if (err != Z_STREAM_END) throw std::runtime_error("load_the_npz_array: inflate failed");
    err = inflateEnd(&d_stream);
    if (err != Z_OK) throw std::runtime_error("load_the_npz_array: inflateEnd failed");

    cnpy::Shape shape;
    size_t word_size;
    bool fortran_order;
    cnpy::parse_npy_header(&buffer_uncompr[0], word_size, shape, fortran_order);

    cnpy::NpyArray array(shape, word_size, fortran_order);

    size_t offset = uncompr_bytes - array.num_bytes();
    memcpy(array.data<unsigned char>(), &buffer_uncompr[0] + offset, array.num_bytes());

    return array;
}
// Helper to memory-map a .npy array within a .npz file
cnpy::NpyArray load_the_npy_mmap(FILE* fp) {
    long data_pos = ftell(fp);
    auto mmap_file = std::make_shared<cnpy::MMapFile>(fileno(fp), "rw");
    unsigned char* buffer = reinterpret_cast<unsigned char*>(const_cast<char*>(mmap_file->data()));
    size_t word_size;
    cnpy::Shape shape;
    bool fortran_order;
    cnpy::parse_npy_header(buffer + data_pos, word_size, shape, fortran_order);
    uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer + data_pos + 8);
    size_t data_offset = data_pos + 10 + header_len;
    return cnpy::NpyArray(shape, word_size, fortran_order, mmap_file, data_offset);
}

// mmap-enabled overload for npz_save (pointer version) removed (duplicate)
cnpy::npz_t cnpy::npz_load(std::string fname, bool use_mmap) {
    FILE* fp = fopen(fname.c_str(), use_mmap ? "rb+" : "rb");

    if (!fp) {
        throw std::runtime_error("npz_load: Error! Unable to open file " + fname + "!");
    }

    cnpy::npz_t arrays;

    while (1) {
        std::vector<char> local_header(30);
        size_t headerres = fread(&local_header[0], sizeof(char), 30, fp);
        if (headerres != 30) throw std::runtime_error("npz_load: failed fread");

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        std::string varname(name_len, ' ');
        size_t vname_res = fread(&varname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len) throw std::runtime_error("npz_load: failed fread");

        // erase the lagging .npy
        varname.erase(varname.end() - 4, varname.end());

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        if (extra_field_len > 0) {
            std::vector<char> buff(extra_field_len);
            size_t efield_res = fread(&buff[0], sizeof(char), extra_field_len, fp);
            if (efield_res != extra_field_len) throw std::runtime_error("npz_load: failed fread");
        }

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        if (compr_method == 0) {
            if (use_mmap) {
                arrays[varname] = load_the_npy_mmap(fp);
                // skip mmap data to advance file pointer past this entry
                fseek(fp, uncompr_bytes, SEEK_CUR);
            } else {
                arrays[varname] = load_the_npy_file(fp);
            }
        } else {
            if (use_mmap) {
                std::cerr << "Warning: npz_load: memory map requested but file '" << fname << "' entry '" << varname
                          << "' is compressed; falling back to memory load" << std::endl;
            }
            arrays[varname] = load_the_npz_array(fp, compr_bytes, uncompr_bytes);
        }
    }

    fclose(fp);
    return arrays;
}

cnpy::NpyArray cnpy::npz_load(std::string fname, std::string varname, bool use_mmap) {
    FILE* fp = fopen(fname.c_str(), use_mmap ? "rb+" : "rb");

    if (!fp) throw std::runtime_error("npz_load: Unable to open file " + fname);

    while (1) {
        std::vector<char> local_header(30);
        size_t header_res = fread(&local_header[0], sizeof(char), 30, fp);
        if (header_res != 30) throw std::runtime_error("npz_load: failed fread");

        // if we've reached the global header, stop reading
        if (local_header[2] != 0x03 || local_header[3] != 0x04) break;

        // read in the variable name
        uint16_t name_len = *(uint16_t*)&local_header[26];
        std::string vname(name_len, ' ');
        size_t vname_res = fread(&vname[0], sizeof(char), name_len, fp);
        if (vname_res != name_len) throw std::runtime_error("npz_load: failed fread");
        vname.erase(vname.end() - 4, vname.end()); // erase the lagging .npy

        // read in the extra field
        uint16_t extra_field_len = *(uint16_t*)&local_header[28];
        fseek(fp, extra_field_len, SEEK_CUR); // skip past the extra field

        uint16_t compr_method = *reinterpret_cast<uint16_t*>(&local_header[0] + 8);
        uint32_t compr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0] + 18);
        uint32_t uncompr_bytes = *reinterpret_cast<uint32_t*>(&local_header[0] + 22);

        if (vname == varname) {
            NpyArray array;
            if (use_mmap && compr_method == 0) {
                array = load_the_npy_mmap(fp);
            } else if (compr_method == 0) {
                array = load_the_npy_file(fp);
            } else {
                if (use_mmap) {
                    std::cerr << "Warning: npz_load: memory map requested but file '" << fname << "' entry '" << varname
                              << "' is compressed; falling back to memory load" << std::endl;
                }
                array = load_the_npz_array(fp, compr_bytes, uncompr_bytes);
            }
            fclose(fp);
            return array;
        } else {
            // skip past the data
            uint32_t size = *(uint32_t*)&local_header[22];
            fseek(fp, size, SEEK_CUR);
        }
    }

    fclose(fp);

    // if we get here, we haven't found the variable in the file
    throw std::runtime_error("npz_load: Variable name " + varname + " not found in " + fname);
}

cnpy::NpyArray cnpy::npy_load(std::string fname, bool use_mmap) {

    if (!use_mmap) {
        FILE* fp = fopen(fname.c_str(), "rb");

        if (!fp) throw std::runtime_error("npy_load: Unable to open file " + fname);

        NpyArray arr = load_the_npy_file(fp);

        fclose(fp);
        return arr;
    } else {
#ifdef __unix__
        // Open and memory-map the file (read-write mode)
        std::shared_ptr<MMapFile> mmap_file = std::make_shared<MMapFile>(fname, "rw");
        // Obtain raw pointer to the mapped region
        unsigned char* buffer = reinterpret_cast<unsigned char*>(const_cast<char*>(mmap_file->data()));
        // Parse the header from the mapped memory
        size_t word_size;
        Shape shape;
        bool fortran_order;
        cnpy::parse_npy_header(buffer, word_size, shape, fortran_order);
        // Header length is stored at offset 8 (little-endian uint16)
        uint16_t header_len = *reinterpret_cast<uint16_t*>(buffer + 8);
        size_t data_offset = 10 + header_len; // 10 bytes before header data
        // Construct an NpyArray that references the mmap region
        cnpy::NpyArray arr(shape, word_size, fortran_order, mmap_file, data_offset);
        return arr;
#else
        stderr << "mmap not supported – fallback to regular load" << std::endl;
        return npy_load(fname, false);
#endif
    }
}

// Implementation of new_npz_mmap
cnpy::npz_t cnpy::new_npz_mmap(std::string filename, const std::vector<ShapeAndType>& _shapes, bool _fortran_order) {
    if (_shapes.empty()) return npz_t();
    bool first = true;
    for (const auto& st : _shapes) {
        const auto& shape = st.shape;
        size_t nvals = 1;
        for (size_t dim : shape) nvals *= dim;
        std::string mode = first ? "w" : "a";
        first = false;
        const std::type_info& tinfo = st.type_info;
        if (tinfo == typeid(int)) {
            std::vector<int> data(nvals);
            npz_save<int>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(float)) {
            std::vector<float> data(nvals);
            npz_save<float>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(double)) {
            std::vector<double> data(nvals);
            npz_save<double>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(long double)) {
            std::vector<long double> data(nvals);
            npz_save<long double>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(char)) {
            std::vector<char> data(nvals);
            npz_save<char>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(short)) {
            std::vector<short> data(nvals);
            npz_save<short>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(long)) {
            std::vector<long> data(nvals);
            npz_save<long>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(long long)) {
            std::vector<long long> data(nvals);
            npz_save<long long>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(unsigned char)) {
            std::vector<unsigned char> data(nvals);
            npz_save<unsigned char>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(unsigned short)) {
            std::vector<unsigned short> data(nvals);
            npz_save<unsigned short>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(unsigned int)) {
            std::vector<unsigned int> data(nvals);
            npz_save<unsigned int>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(unsigned long)) {
            std::vector<unsigned long> data(nvals);
            npz_save<unsigned long>(filename, st.name, data.data(), shape, mode, false);
        } else if (tinfo == typeid(unsigned long long)) {
            std::vector<unsigned long long> data(nvals);
            npz_save<unsigned long long>(filename, st.name, data.data(), shape, mode, false);
        } else {
            throw std::runtime_error("new_npz_mmap: unsupported type for variable " + st.name);
        }
    }
    // Return memory-mapped arrays
    auto arrays = npz_load(filename, true);
    for (auto& kv : arrays) {
        kv.second.fortran_order = _fortran_order;
    }
    return arrays;
}
