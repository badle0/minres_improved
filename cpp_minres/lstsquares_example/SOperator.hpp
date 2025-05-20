// SOperator.hpp

#ifndef S_OPERATOR_HPP
#define S_OPERATOR_HPP

#include "LstsquaresVector.hpp"
#include "../helpers/helper_functions.hpp"

/*!
 * @brief An operator that applies a CSR matrix S (size 2n x 2n) to a vector.
 */
class SOperator {
public:
    //! Constructor that takes a CSRMatrix (already loaded) of size (2n x 2n).
    explicit SOperator(const CSRMatrix & S_csr)
        : S_(S_csr)
    {
        // Optionally, verify it's square etc.
        if(S_.num_rows != S_.num_cols) {
            throw std::runtime_error("SOperator: CSRMatrix is not square!");
        }
    }

    //! Apply the matrix S to x, storing in y. Both x,y must be size = S_.num_rows.
    void Apply(const LstsquaresVector & x, LstsquaresVector & y) const {
        // Check sizes
        if(x.size != S_.num_rows || y.size != S_.num_rows) {
            throw std::runtime_error("SOperator: dimension mismatch in Apply().");
        }

        // Standard CSR matrix-vector multiply:
        //  For row i in [0..num_rows-1],
        //    y[i] = sum_{j in [indptr[i]..indptr[i+1])} data[j] * x[ indices[j] ]
        for(int i = 0; i < S_.num_rows; i++) {
            double sum = 0.0;
            int start = S_.indptr[i];
            int end   = S_.indptr[i+1];
            for(int jj = start; jj < end; ++jj) {
                int col = S_.indices[jj];
                double val = S_.data[jj];
                sum += val * x[col];
            }
            y[i] = sum;
        }
    }

private:
    CSRMatrix S_; // The loaded matrix in CSR format.
};

#endif // S_OPERATOR_HPP
