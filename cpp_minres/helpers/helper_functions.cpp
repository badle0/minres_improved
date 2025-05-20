#include "helper_functions.hpp"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstring>  // for std::memcpy

CSRMatrix readA_sparse(const std::string & filename, DataType dtype)
{
    CSRMatrix csr;

    std::ifstream fin(filename, std::ios::binary);
    if(!fin.is_open()) {
        throw std::runtime_error("readA_sparse: Cannot open file " + filename);
    }

    // Read 5 ints: num_rows, num_cols, nnz, outS, innS
    int outS = 0, innS = 0;
    fin.read(reinterpret_cast<char*>(&csr.num_rows), sizeof(int));
    fin.read(reinterpret_cast<char*>(&csr.num_cols), sizeof(int));
    fin.read(reinterpret_cast<char*>(&csr.nnz),      sizeof(int));
    fin.read(reinterpret_cast<char*>(&outS),         sizeof(int));
    fin.read(reinterpret_cast<char*>(&innS),         sizeof(int));

    if(!fin.good()) {
        throw std::runtime_error("readA_sparse: Error reading header from " + filename);
    }

    // ---------------------------------------------------------------------
    // 1) Compute the expected file size in bytes:
    //    - 5 header ints
    //    - data array: nnz * (sizeof float or double)
    //    - indptr array: outS ints
    //    - indices array: nnz ints
    // ---------------------------------------------------------------------
    size_t expectedSize = sizeof(int)*5; // for the 5 header ints

    size_t len_data = 0; // size in bytes of each data element
    if (dtype == DataType::Float32) {
        len_data = sizeof(float);
    } else {
        len_data = sizeof(double);
    }
    expectedSize += static_cast<size_t>(csr.nnz) * len_data;         // data
    expectedSize += static_cast<size_t>(outS) * sizeof(int);         // indptr
    expectedSize += static_cast<size_t>(csr.nnz) * sizeof(int);      // indices

    // Now let's check the *actual* file size on disk
    // We do this by opening the same file in a separate ifstream in "ate" mode
    // or we can do it with fin directly (but must restore fin's position).
    std::streampos currentPos = fin.tellg(); // store where we are after header
    fin.seekg(0, std::ios::end);
    std::streampos actualSize = fin.tellg();

    // restore position (so we can read the data array next)
    fin.seekg(currentPos, std::ios::beg);

    if (actualSize < 0) {
        throw std::runtime_error("readA_sparse: Error obtaining file size for " + filename);
    }
    size_t fileSize = static_cast<size_t>(actualSize);

    // Compare
    if (fileSize != expectedSize) {
        std::cerr << "WARNING: For file " << filename << ":\n"
                  << "  expectedSize = " << expectedSize
                  << ", actual size = " << fileSize << "\n"
                  << "This could indicate truncated file or mismatch in header.\n";
        // You could throw or just warn:
        // throw std::runtime_error("Size mismatch, possible truncated file.");
    }

    // ---------------------------------------------------------------------
    // 2) Now read the data arrays as usual
    // ---------------------------------------------------------------------
    csr.data.resize(csr.nnz);
    if (dtype == DataType::Float32) {
        std::vector<float> tmp(csr.nnz);
        fin.read(reinterpret_cast<char*>(tmp.data()), csr.nnz * sizeof(float));
        for(int i = 0; i < csr.nnz; i++) {
            csr.data[i] = static_cast<double>(tmp[i]);
        }
    } else {
        fin.read(reinterpret_cast<char*>(csr.data.data()), csr.nnz * sizeof(double));
    }
    if(!fin.good()) {
        throw std::runtime_error("readA_sparse: Error reading data array from " + filename);
    }

    csr.indptr.resize(outS);
    fin.read(reinterpret_cast<char*>(csr.indptr.data()), outS * sizeof(int));
    if(!fin.good()) {
        throw std::runtime_error("readA_sparse: Error reading indptr from " + filename);
    }

    csr.indices.resize(csr.nnz);
    fin.read(reinterpret_cast<char*>(csr.indices.data()), csr.nnz * sizeof(int));
    if(!fin.good()) {
        throw std::runtime_error("readA_sparse: Error reading indices from " + filename);
    }

    fin.close();
    return csr;
}


LstsquaresVector get_vector_from_source(const std::string & filename, DataType dtype)
{
    std::ifstream fin(filename, std::ios::binary);
    if(!fin.is_open()) {
        std::cerr << "get_vector_from_source: File does not exist at " << filename << std::endl;
        return LstsquaresVector(0); // return empty vector
    }

    // We read the entire file into a buffer of float or double, then convert to double in the end.
    fin.seekg(0, std::ios::end);
    std::streampos fileSize = fin.tellg();
    fin.seekg(0, std::ios::beg);

    if(fileSize <= 0) {
        std::cerr << "get_vector_from_source: File is empty or error reading size. " << filename << std::endl;
        return LstsquaresVector(0);
    }

    // Figure out how many elements we have based on dtype:
    // For float32 each element is 4 bytes, for float64 each element is 8 bytes.
    size_t numElements = 0;
    if(dtype == DataType::Float32) {
        numElements = static_cast<size_t>(fileSize / sizeof(float));
    } else {
        numElements = static_cast<size_t>(fileSize / sizeof(double));
    }

    LstsquaresVector vec(static_cast<int>(numElements));
    if(numElements == 0) {
        // no data
        return vec;
    }

    if (dtype == DataType::Float32) {
        std::vector<float> tmp(numElements);
        fin.read(reinterpret_cast<char*>(tmp.data()), numElements * sizeof(float));
        if(!fin.good()) {
            std::cerr << "get_vector_from_source: Error reading float data from " << filename << std::endl;
            return LstsquaresVector(0);
        }
        for(size_t i = 0; i < numElements; i++) {
            vec[i] = static_cast<double>(tmp[i]);
        }
    } else {
        // Read as double directly
        fin.read(reinterpret_cast<char*>(vec.vals), numElements * sizeof(double));
        if(!fin.good()) {
            std::cerr << "get_vector_from_source: Error reading double data from " << filename << std::endl;
            return LstsquaresVector(0);
        }
    }

    fin.close();
    return vec;
}
