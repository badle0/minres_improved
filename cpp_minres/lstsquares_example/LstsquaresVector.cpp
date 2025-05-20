// LstsquaresVector.cpp
// tminres is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License, version 2.1.

#include "LstsquaresVector.hpp"

// Constructor: allocate the array for 'size' elements.
LstsquaresVector::LstsquaresVector(int size_):
    size(size_)
{
    assert(size > 0);
    vals = new double[size];
}

LstsquaresVector::LstsquaresVector(const LstsquaresVector & other)
    : size(other.size)
{
    vals = new double[size];
    std::copy(other.vals, other.vals + size, vals);
}


// Destructor: free the allocated memory.
LstsquaresVector::~LstsquaresVector()
{
    delete[] vals;
}

// Overload operator= to set all entries equal to a given double value.
LstsquaresVector & LstsquaresVector::operator=(const double & val)
{
    std::fill(vals, vals + size, val);
    return *this;
}

// Overload operator= to copy all elements from another SimpleVector.
LstsquaresVector & LstsquaresVector::operator=(const LstsquaresVector & RHS)
{
    assert(size == RHS.size);
    std::copy(RHS.vals, RHS.vals + RHS.size, vals);
    return *this;
}

// Multiply all elements by a scalar.
void LstsquaresVector::Scale(const double & val)
{
    for (double* it = vals; it != vals + size; ++it)
        (*it) *= val;
}

// Create a new SimpleVector with the same size.
LstsquaresVector* LstsquaresVector::Clone()
{
    return new LstsquaresVector(size);
}

// Overload operator[] for non-const access.
double & LstsquaresVector::operator[](const int i)
{
    assert(i >= 0 && i < size);
    return vals[i];
}

// Overload operator[] for const access.
const double & LstsquaresVector::operator[](const int i) const
{
    assert(i >= 0 && i < size);
    return vals[i];
}

// Safe element access: returns 0.0 if index is out-of-bound.
const double LstsquaresVector::at(const int i) const
{
    if (i < 0 || i >= size)
        return 0.0;
    return vals[i];
}

// Randomize vector entries (uniformly in [-1, 1]) and then normalize the vector.
void LstsquaresVector::Randomize(int seed)
{
    srand(seed);
    for (double* it = vals; it != vals + size; ++it)
        (*it) = 2.0 * static_cast<double>(rand()) / static_cast<double>(RAND_MAX) - 1.0;

    double norm2 = InnerProduct(*this, *this);
    Scale(1.0 / std::sqrt(norm2));
}

// Print the vector to the given output stream.
void LstsquaresVector::Print(std::ostream & os)
{
    for (double* it = vals; it != vals + size; ++it)
        os << *it << "\t ";
    os << "\n";
}

// ---- Free functions for vector arithmetic ----

// Compute: result = v1 + c2*v2
void add(const LstsquaresVector & v1, const double & c2, const LstsquaresVector & v2, LstsquaresVector & result)
{
    assert(v1.size == v2.size && result.size == v1.size);
    for (int i = 0; i < result.size; i++)
        result.vals[i] = v1.vals[i] + c2 * v2.vals[i];
}

// Compute: result = c1*v1 + c2*v2
void add(const double & c1, const LstsquaresVector & v1, const double & c2, const LstsquaresVector & v2, LstsquaresVector & result)
{
    assert(v1.size == v2.size && result.size == v1.size);
    for (int i = 0; i < result.size; i++)
        result.vals[i] = c1 * v1.vals[i] + c2 * v2.vals[i];
}

// Compute: result = alpha*(v1 + v2)
void add(const double & alpha, const LstsquaresVector & v1, const LstsquaresVector & v2, LstsquaresVector & result)
{
    assert(v1.size == v2.size && result.size == v1.size);
    for (int i = 0; i < result.size; i++)
        result.vals[i] = alpha * (v1.vals[i] + v2.vals[i]);
}

// Compute: result = v1 + v2 + v3
void add(const LstsquaresVector & v1, const LstsquaresVector & v2, const LstsquaresVector & v3, LstsquaresVector & result)
{
    assert(v1.size == v2.size && v2.size == v3.size && result.size == v1.size);
    for (int i = 0; i < result.size; i++)
        result.vals[i] = v1.vals[i] + v2.vals[i] + v3.vals[i];
}

// Compute: result = v1 - v2
void subtract(const LstsquaresVector & v1, const LstsquaresVector & v2, LstsquaresVector & result)
{
    assert(v1.size == v2.size && result.size == v1.size);
    for (int i = 0; i < result.size; i++)
        result.vals[i] = v1.vals[i] - v2.vals[i];
}

// Compute the inner (dot) product of v1 and v2.
double InnerProduct(const LstsquaresVector & v1, const LstsquaresVector & v2)
{
    assert(v1.size == v2.size);
    double result = 0.0;
    for (int i = 0; i < v1.size; i++)
        result += v1.vals[i] * v2.vals[i];
    return result;
}
