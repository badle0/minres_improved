# basic_tests.py: Create different systems (SPD, nonsymmetric indefinite, etc.)
# and test various interative linear solvers: custom/scipy minres/BiCGSTAB/gmres.

import sys
import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import minres
from scipy.sparse import csc_matrix
import time
import argparse

# Custom input modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../lib'))
sys.path.insert(1, lib_path)

from minres import MINRESSparse
from gmres import GMRESSparse
import helper_functions as hf


# FUNCTION COPIED FROM HONGYI PYTHON FILE: create_tridiagA_andS.py
def random_tridiagonal_matrix(n, seed=42):
    """
    Generates an n x n tridiagonal matrix in CSR format, where
    the main diagonal and off-diagonals contain random values in [0,1].

    Parameters:
    -----------
    n : int
        Size of the matrix (n x n).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns:
    --------
    A : scipy.sparse.csr_matrix
        The resulting tridiagonal matrix.
    """
    # Set the random seed for reproducibility
    np.random.seed(seed)

    # Generate random values for main diagonal
    main_diag = np.random.rand(n)

    # Generate random values for off-diagonals
    off_diag = np.random.rand(n - 1)

    # Construct the sparse tridiagonal matrix
    A = sp.diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format='csr'
    )

    # Find the largest eigenvalue in magnitude
    eigenvalues, _ = eigsh(A, k=1, which='LM')
    A_normalized = A / np.abs(eigenvalues[0])

    return A_normalized


# FUNCTION COPIED FROM HONGYI PYTHON FILE: create_tridiagA_andS.py
def spaugment(A, c):
    """
    Constructs a sparse augmented matrix S from matrix A and scalar c.

    Parameters:
        A (scipy.sparse.csr_matrix): Input matrix of size n x n.
        c (float): Scalar to multiply with the identity matrix.

    Returns:
        scipy.sparse.csr_matrix: Augmented matrix S of size (n + n) x (n + n).
    """
    n, m = A.shape
    I = sp.identity(n, format='csr')  # Identity matrix of size n
    Z = sp.csr_matrix((n, n))  # Zero matrix of size n

    top = sp.hstack([c * I, A])
    bottom = sp.hstack([A.transpose(), Z])
    S = sp.vstack([top, bottom])
    # S:
    # [c*I A]
    # [A^T 0]
    return S


def generate_indefinite_nonsymmetric_matrix(size, density=0.05, seed=42):
    """
    Generates an indefinite, nonsymmetric sparse matrix in CSR format.

    Parameters:
    -----------
    size : int
        Size of the matrix (size x size).
    density : float
        Density of the non-zero elements (between 0 and 1).
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns:
    --------
    A : scipy.sparse.csr_matrix
        The resulting indefinite, nonsymmetric sparse matrix.
    """
    np.random.seed(seed)

    # Generate a random sparse matrix with given density
    A = sp.random(size, size, density=density, format='csr', data_rvs=np.random.rand)

    # Ensure matrix is nonsymmetric
    A = A - sp.diags(A.diagonal())  # Remove diagonal elements to enhance nonsymmetry

    # Check for indefiniteness
    eigenvalues, _ = eigs(A, k=2, which='LR')  # Get a few largest real parts of eigenvalues

    iteration = 0
    while not (np.any(eigenvalues.real > 0) and np.any(eigenvalues.real < 0)) and iteration < 10:
        # Apply sparse perturbation
        perturb_indices = np.random.choice(size * size, size, replace=False)
        perturb_data = np.random.rand(size) - 0.5
        perturb_matrix = sp.csr_matrix((perturb_data, (perturb_indices // size, perturb_indices % size)), shape=(size, size))

        A += perturb_matrix  # Apply sparse perturbation
        eigenvalues, _ = eigs(A, k=2, which='LR')
        iteration += 1

    return A


def identity_matrix_test(dim):
    print("IDENTITY MATRIX")
    A = np.eye(dim) # A = I (Size: dim*dim)
    for i in range(dim):
        A[i,i]=1
    b = np.arange(dim)
    print("A=\n", A) # A = I
    print("b=", b) # b = [0 1 2 ... dim-1]

    print("\nSCIPY MINRES")
    x_sol_mres, exitCode = minres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_mres)
    
    print("\nnCUSTOM MINRES")
    MR = MINRESSparse(A)
    x_sol_mr, res_arr_mr = MR.minres(b, np.ones(b.shape))
    print("x_sol_mr=", x_sol_mr)
    print("res_arr_mr=", res_arr_mr, "\n")

    print("\nSCIPY GMRES")
    x, exitCode = gmres(A, b)
    print("exitcode=",exitCode)
    print("x_sol_gmres=",x)

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    x_sol_gm, res_arr_gm = GM.gmres(b, np.ones(dim))
    print("x_sol_gm=", x_sol_gm)
    print("res_arr_gm=", res_arr_gm, "\n")


def symmetric_positive_definite_test(dim):
    print("\nSYMMETRIC POSITIVE DEFINITE MATRIX")

    # Generate a random symmetric positive definite matrix
    np.random.seed(42)
    A = np.random.rand(dim, dim)
    A = np.dot(A, A.T) + dim * np.eye(dim)  # Ensure positive definiteness

    b = np.arange(dim)
    print("A=\n", pd.DataFrame(A))
    print("b=", b)

    # Solving Ax = b
    print("\nSCIPY MINRES")
    x_sol_mres, exitCode = minres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_mres)
    print("Ax=", A.dot(x_sol_mres))

    print("\nCUSTOM MINRES")
    MR = MINRESSparse(A)
    x_sol_mr, res_arr_mr = MR.minres(b, np.ones(b.shape))
    print("x_sol_mr=", x_sol_mr)
    print("res_arr_mr=", res_arr_mr)
    print("Ax=", A.dot(x_sol_mr))

    print("\nSCIPY GMRES")
    x_sol_gmres, exitCode = gmres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_gmres)
    print("Ax=", A.dot(x_sol_gmres))

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    x_sol_gm, res_arr_gm = GM.gmres(b, np.ones(dim))
    print("x=", x_sol_gm)
    print("res_arr_gm=", res_arr_gm)
    print("Ax=", A.dot(x_sol_gm))


def symmetric_inpositive_definite_test(dim):
    print("\nSYMMETRIC INDEFINITE")
    tri_A = random_tridiagonal_matrix(dim)
    b=np.arange(dim)
    S = spaugment(tri_A, 1)
    d = np.append(b, np.zeros(dim))
    # Ax = b 
    # SPAUGMENTED BECOMES:
    # Sy = d
    print("SOLVING Sy=d for spaugmented system...")
    print("S=\n",pd.DataFrame(S.toarray()))
    print("d=", d) # b = [0,1,2,..,dim-1]
    

    print("\nSCIPY MINRES")
    y_sol_mres, exitCode = minres(S, d)
    print('exitcode=', exitCode)
    print("y=", y_sol_mres)
    print("Sy=",S.dot(y_sol_mres))

    print("\nCUSTOM MINRES")
    MR = MINRESSparse(S)
    y_sol_mr, res_arr_mr = MR.minres(d, np.ones(d.shape))
    print("y_sol_mr=", y_sol_mr)
    print("res_arr_mr=", res_arr_mr)
    print("Sy=",S.dot(y_sol_mr))

    print("\nSCIPY GMRES")
    y_sol_gmres, exitCode = gmres(S, d)
    print('exitcode=', exitCode)
    print("y=", y_sol_gmres)
    print("Sy=",S.dot(y_sol_gmres))

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(S)
    y_sol_gmres, res_arr_gm = GM.gmres(d, np.ones(dim * 2))
    print("y=", y_sol_gmres)
    print("res_arr_gm=", res_arr_gm)
    print("Sy=",S.dot(y_sol_gmres))


def nonsymmetric_indefinite_test(dim):
    print("\nNONSYMMETRIC INDEFINITE MATRIX")
    A = generate_indefinite_nonsymmetric_matrix(dim)
    b = np.arange(dim)
    print("A=\n",pd.DataFrame(A.toarray()))
    print("b=",b)

    print("\nSCIPY GMRES")
    x_sol_gmres, exitCode = gmres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_gmres)
    print("Ax=",A.dot(x_sol_gmres))

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    x_sol_gmres, res_arr_gm = GM.gmres(b, np.ones(dim))
    print("x=", x_sol_gmres)
    print("res_arr_gm=", res_arr_gm)
    print("Ax=",A.dot(x_sol_gmres))

    print("\nSCIPY MINRES (should FAIL)")
    x_sol_mres, exitCode = minres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_mres)
    print("Ax=",A.dot(x_sol_mres))


    print("\nCUSTOM MINRES (should FAIL)")
    MR = MINRESSparse(A)
    x_sol_mr, res_arr_mr = MR.minres(b, np.ones(b.shape))
    print("x_sol_mr=", x_sol_mr)
    print("res_arr_mr=", res_arr_mr)
    print("Ax=",A.dot(x_sol_mr))


def main(dim):
    print("--== BEGINNING TESTING ==--\n")
    identity_matrix_test(dim)
    symmetric_positive_definite_test(dim)
    symmetric_inpositive_definite_test(dim)
    nonsymmetric_indefinite_test(dim)

if __name__ == "__main__":
    main(8)
    
# do diagonal preconditioners. incomplete LU
# should not take more than one iteration for GMRES to converge with Identity matrix