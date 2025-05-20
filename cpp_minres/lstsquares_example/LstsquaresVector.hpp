#ifndef LSTSQUARESVECTOR_HPP
#define LSTSQUARESVECTOR_HPP

#include <iostream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdlib>

//! A simple dense vector class for MINRES operations.
class LstsquaresVector {
public:
    // Public data members (for direct access by friend functions)
    int size;       //!< Number of elements in the vector.
    double* vals;   //!< Pointer to the array of vector values.

    //! Constructor. Allocates space for a vector of the given size.
    LstsquaresVector(int size_);

    // Add a copy constructor for deep copying.
    LstsquaresVector(const LstsquaresVector & other);

    //! Destructor.
    ~LstsquaresVector();

    //! Overload assignment operator to set all entries to a given double value.
    LstsquaresVector & operator=(const double & val);

    //! Overload assignment operator to copy values from another SimpleVector.
    LstsquaresVector & operator=(const LstsquaresVector & RHS);

    //! Scale (multiply) all elements by a scalar (in-place).
    void Scale(const double & val);

    //! Create a new vector with the same structure as this one (values not initialized).
    LstsquaresVector* Clone();

    //! Element access (non-const). Returns reference to element i.
    double & operator[](const int i);

    //! Element access (const). Returns const reference to element i.
    const double & operator[](const int i) const;

    //! Safe element access. Returns 0.0 if the index is out-of-bound.
    const double at(const int i) const;

    //! Randomize the vector entries (using the given seed) and then normalize the vector.
    void Randomize(int seed);

    //! Print the vector elements to the provided output stream.
    void Print(std::ostream & os);
};

// Free (friend) functions for vector arithmetic operations:

//! Compute: result = v1 + c2*v2
void add(const LstsquaresVector & v1, const double & c2, const LstsquaresVector & v2, LstsquaresVector & result);

//! Compute: result = c1*v1 + c2*v2
void add(const double & c1, const LstsquaresVector & v1, const double & c2, const LstsquaresVector & v2, LstsquaresVector & result);

//! Compute: result = alpha*(v1 + v2)
void add(const double & alpha, const LstsquaresVector & v1, const LstsquaresVector & v2, LstsquaresVector & result);

//! Compute: result = v1 + v2 + v3
void add(const LstsquaresVector & v1, const LstsquaresVector & v2, const LstsquaresVector & v3, LstsquaresVector & result);

//! Compute: result = v1 - v2
void subtract(const LstsquaresVector & v1, const LstsquaresVector & v2, LstsquaresVector & result);

//! Compute the inner (dot) product of v1 and v2.
double InnerProduct(const LstsquaresVector & v1, const LstsquaresVector & v2);

#endif // LSTSQUARESVECTOR_HPP
