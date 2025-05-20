#ifndef LSTSQUARESOPERATOR_HPP
#define LSTSQUARESOPERATOR_HPP

#include <vector>
#include <cassert>
#include "LstsquaresVector.hpp"  // Your vector class (formerly SimpleVector)

/*!
 * @brief LstsquaresOperator implements the augmented least-squares matrix S.
 *
 * The augmented matrix S is defined as:
 *
 *   S = [ c*I, A ;
 *         A^T, 0 ]
 *
 * where:
 *  - c is a scalar,
 *  - A is an n-by-n tridiagonal matrix,
 *  - The overall system size is 2n.
 *
 * Given an input vector x of size 2n, partitioned as:
 *  x = [ x_top ; x_bottom ],
 *
 * the operator computes:
 *  y_top = c*x_top + A*x_bottom,
 *  y_bottom = A^T*x_top.
 *
 * The tridiagonal matrix A is represented by its three diagonals:
 *  - d: main diagonal (length n),
 *  - l: sub-diagonal (length n-1),
 *  - u: super-diagonal (length n-1).
 */
class LstsquaresOperator {
public:
    /// Constructor.
    /// @param n The dimension of A (so A is n x n).
    /// @param c The scalar multiplier for the identity in the top-left block.
    /// By default, A is set with d[i]=2.0, l[i]=u[i]=-1.0.
    LstsquaresOperator(int n, double c)
        : n_(n), c_(c), d(n, 2.0), l(n - 1, -1.0), u(n - 1, -1.0) { }

    /// Alternate constructor to allow custom diagonals.
    LstsquaresOperator(int n, double c,
                        const std::vector<double>& d,
                        const std::vector<double>& l,
                        const std::vector<double>& u)
        : n_(n), c_(c), d(d), l(l), u(u)
    {
        assert(d.size() == static_cast<size_t>(n_));
        assert(l.size() == static_cast<size_t>(n_ - 1));
        assert(u.size() == static_cast<size_t>(n_ - 1));
    }

    /// Apply the operator S to the vector x, storing the result in y.
    /// x and y must both have size 2*n.
    void Apply(const LstsquaresVector & x, LstsquaresVector & y) const {
        // Ensure the input vectors have the expected size.
        assert(x.size == 2 * n_);
        assert(y.size == 2 * n_);

        // Partition x: x_top is x[0...n-1], x_bottom is x[n...2*n-1].
        // Compute y_top = c*x_top + A*x_bottom.
        for (int i = 0; i < n_; i++) {
            double Ax_bottom_i = 0.0;
            int offset = n_;  // Starting index for the bottom half.
            // Main diagonal contribution:
            Ax_bottom_i += d[i] * x[offset + i];
            // Sub-diagonal contribution (if applicable):
            if (i > 0)
                Ax_bottom_i += l[i - 1] * x[offset + i - 1];
            // Super-diagonal contribution (if applicable):
            if (i < n_ - 1)
                Ax_bottom_i += u[i] * x[offset + i + 1];

            y[i] = c_ * x[i] + Ax_bottom_i;
        }

        // Compute y_bottom = A^T * x_top.
        // For A^T, the main diagonal remains the same.
        // The sub-diagonal of A becomes the super-diagonal of A^T and vice versa.
        for (int i = 0; i < n_; i++) {
            double ATx_top_i = 0.0;
            // Main diagonal:
            ATx_top_i += d[i] * x[i];
            // In A^T, the element above the main diagonal comes from l of A.
            if (i < n_ - 1)
                ATx_top_i += l[i] * x[i + 1];
            // And the element below the main diagonal comes from u of A.
            if (i > 0)
                ATx_top_i += u[i - 1] * x[i - 1];

            y[n_ + i] = ATx_top_i;
        }
    }

private:
    int n_;             // Dimension of A (n x n).
    double c_;          // Scalar multiplier for the identity in the top-left block.
    std::vector<double> d; // Main diagonal of A.
    std::vector<double> l; // Sub-diagonal of A (length n-1).
    std::vector<double> u; // Super-diagonal of A (length n-1).
};

#endif // LSTSQUARESOPERATOR_HPP
