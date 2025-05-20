#ifndef HELPER_FUNCTIONS_HPP
#define HELPER_FUNCTIONS_HPP

#include <string>
#include <vector>
#include "lstsquares_example/LstsquaresVector.hpp"

/// Enumeration for selecting either float32 or float64 read.
enum class DataType {
    Float32,
    Float64
};

/// A minimal struct for storing a CSR (compressed sparse row) matrix.
struct CSRMatrix {
    int num_rows = 0;
    int num_cols = 0;
    int nnz = 0;  // number of non-zeros

    // For CSR:
    //   data.size() == nnz
    //   indices.size() == nnz
    //   indptr.size() == num_rows + 1 or outS (depending on your file structure).
    // In your Python code, it was read as (data, indices, indptr).
    std::vector<double> data;     // values of nonzero elements
    std::vector<int>    indices;  // column indices
    std::vector<int>    indptr;   // row pointer

    // If needed, you can add methods for applying this matrix, etc.
};

/// \brief Reads a sparse CSR matrix from a custom binary file (matching your Python logic).
///
/// \param filename  Path to the binary file.
/// \param dtype     Specify whether data is stored as float32 or float64 in the file.
///
/// The binary file structure is assumed to match your Python code:
///   1) int num_rows
///   2) int num_cols
///   3) int nnz
///   4) int outS
///   5) int innS
///   6) data (nnz elements of float/double)
///   7) indptr (outS elements of int)
///   8) indices (nnz elements of int)
///
/// \return A CSRMatrix with the stored data converted to double.
CSRMatrix readA_sparse(const std::string & filename, DataType dtype);

/// \brief Reads a raw binary vector of floats or doubles from file into LstsquaresVector.
///
/// \param filename  Path to the .bin file containing the raw vector.
/// \param dtype     Specify whether data is stored as float32 or float64.
/// \return A LstsquaresVector containing the data in double precision.
///         If the file is empty or missing, returns an empty vector (size=0).
LstsquaresVector get_vector_from_source(const std::string & filename, DataType dtype);

#endif // HELPER_FUNCTIONS_HPP
