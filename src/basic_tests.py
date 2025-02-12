# basic_tests.py: Create different systems (SPD, nonsymmetric indefinite, etc.)
# and test various iterative linear solvers: custom/scipy minres/BiCGSTAB/gmres.

import sys
import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import pandas as pd
from scipy.sparse.linalg import eigsh, eigs
import time
import argparse

# Custom input modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../lib'))
sys.path.insert(1, lib_path)

from minres import MINRESSparse
from gmres import GMRESSparse
from bicgstab import BiCGSTABSparse
import helper_functions as hf


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
    np.random.seed(seed)
    main_diag = np.random.rand(n)
    off_diag = np.random.rand(n - 1)
    A = sp.diags(
        diagonals=[off_diag, main_diag, off_diag],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format='csr'
    )
    eigenvalues, _ = eigsh(A, k=1, which='LM')
    A_normalized = A / np.abs(eigenvalues[0])
    return A_normalized


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
    I = sp.identity(n, format='csr')
    Z = sp.csr_matrix((n, n))
    top = sp.hstack([c * I, A])
    bottom = sp.hstack([A.transpose(), Z])
    S = sp.vstack([top, bottom])
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
    A = sp.random(size, size, density=density, format='csr', data_rvs=np.random.rand)
    A = A - sp.diags(A.diagonal())
    eigenvalues, _ = eigs(A, k=2, which='LR')
    iteration = 0
    while not (np.any(eigenvalues.real > 0) and np.any(eigenvalues.real < 0)) and iteration < 10:
        perturb_indices = np.random.choice(size * size, size, replace=False)
        perturb_data = np.random.rand(size) - 0.5
        perturb_matrix = sp.csr_matrix((perturb_data, (perturb_indices // size, perturb_indices % size)), shape=(size, size))
        A += perturb_matrix
        eigenvalues, _ = eigs(A, k=2, which='LR')
        iteration += 1
    return A


def identity_matrix_test(dim):
    print("IDENTITY MATRIX")
    A = np.eye(dim)
    b = np.arange(dim)
    print("A=\n", A)
    print("b=", b)

    print("\nSCIPY MINRES")
    x_sol_mres, exitCode = sp.linalg.minres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_mres)
    
    print("\nCUSTOM MINRES")
    MR = MINRESSparse(A)
    x_sol_mr, res_arr_mr = MR.minres(b, np.ones(b.shape))
    print("x_sol_mr=", x_sol_mr)
    print("res_arr_mr=", res_arr_mr, "\n")

    print("\nSCIPY GMRES")
    x_sol_gmres, exitCode = sp.linalg.gmres(A, b)
    print("exitcode=", exitCode)
    print("x_sol_gmres=", x_sol_gmres)

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    x_sol_gm, res_arr_gm = GM.gmres(b, np.ones(dim))
    print("x_sol_gm=", x_sol_gm)
    print("res_arr_gm=", res_arr_gm, "\n")

    print("\nSCIPY BiCGSTAB")
    x_sol_bicgstab, exitCode = sp.linalg.bicgstab(A, b)
    print("exitcode=", exitCode)
    print("x_sol_bicgstab=", x_sol_bicgstab)
    
    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(A)
    x_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(b, np.ones(dim))
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab, "\n")


def symmetric_positive_definite_test(dim):
    print("\nSYMMETRIC POSITIVE DEFINITE MATRIX")
    np.random.seed(42)
    A = np.random.rand(dim, dim)
    A = np.dot(A, A.T) + dim * np.eye(dim)
    b = np.arange(dim)
    print("A=\n", pd.DataFrame(A))
    print("b=", b)

    print("\nSCIPY MINRES")
    x_sol_mres, exitCode = sp.linalg.minres(A, b)
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
    x_sol_gmres, exitCode = sp.linalg.gmres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_gmres)
    print("Ax=", A.dot(x_sol_gmres))

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    x_sol_gm, res_arr_gm = GM.gmres(b, np.ones(dim))
    print("x=", x_sol_gm)
    print("res_arr_gm=", res_arr_gm)
    print("Ax=", A.dot(x_sol_gm))
    
    print("\nSCIPY BiCGSTAB")
    x_sol_bicgstab, exitCode = sp.linalg.bicgstab(A, b)
    print("exitcode=", exitCode)
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))

    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(A)
    x_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(b, np.ones(dim))
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))


def symmetric_indefinite_test(dim):
    print("\nSYMMETRIC INDEFINITE")
    tri_A = random_tridiagonal_matrix(dim)
    b = np.arange(dim)
    S = spaugment(tri_A, 1)
    d = np.append(b, np.zeros(dim))
    print("SOLVING Sy=d for spaugmented system...")
    print("S=\n", pd.DataFrame(S.toarray()))
    print("d=", d)

    print("\nSCIPY MINRES")
    y_sol_mres, exitCode = sp.linalg.minres(S, d)
    print('exitcode=', exitCode)
    print("y=", y_sol_mres)
    print("Sy=", S.dot(y_sol_mres))

    print("\nCUSTOM MINRES")
    MR = MINRESSparse(S)
    y_sol_mr, res_arr_mr = MR.minres(d, np.ones(d.shape))
    print("y_sol_mr=", y_sol_mr)
    print("res_arr_mr=", res_arr_mr)
    print("Sy=", S.dot(y_sol_mr))

    print("\nSCIPY GMRES")
    y_sol_gmres, exitCode = sp.linalg.gmres(S, d)
    print('exitcode=', exitCode)
    print("y=", y_sol_gmres)
    print("Sy=", S.dot(y_sol_gmres))

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(S)
    y_sol_gmres, res_arr_gm = GM.gmres(d, np.ones(dim * 2))
    print("y=", y_sol_gmres)
    print("res_arr_gm=", res_arr_gm)
    print("Sy=", S.dot(y_sol_gmres))
    
    print("\nSCIPY BiCGSTAB")
    y_sol_bicgstab, exitCode = sp.linalg.bicgstab(S, d, np.ones(dim * 2))
    print("exitcode=", exitCode)
    print("y_sol_bicgstab=", y_sol_bicgstab)
    print("Sy=", S.dot(y_sol_bicgstab))

    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(S)
    y_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(d, np.ones(dim * 2))
    print("y_sol_bicgstab=", y_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Sy=", S.dot(y_sol_bicgstab))


def nonsymmetric_indefinite_test(dim):
    print("\nNONSYMMETRIC INDEFINITE MATRIX")
    A = generate_indefinite_nonsymmetric_matrix(dim)
    b = np.arange(dim)
    print("A=\n", pd.DataFrame(A.toarray()))
    print("b=", b)

    # print("\nSCIPY MINRES (should FAIL)")
    # x_sol_mres, exitCode = sp.linalg.minres(A, b)
    # print('exitcode=', exitCode)
    # print("x=", x_sol_mres)
    # print("Ax=", A.dot(x_sol_mres))

    # print("\nCUSTOM MINRES (should FAIL)")
    # MR = MINRESSparse(A)
    # x_sol_mr, res_arr_mr = MR.minres(b, np.ones(b.shape))
    # print("x_sol_mr=", x_sol_mr)
    # print("res_arr_mr=", res_arr_mr)
    # print("Ax=", A.dot(x_sol_mr))

    print("\nSCIPY GMRES")
    x_sol_gmres, exitCode = sp.linalg.gmres(A, b)
    print('exitcode=', exitCode)
    print("x=", x_sol_gmres)
    print("Ax=", A.dot(x_sol_gmres))

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    x_sol_gmres, res_arr_gm = GM.gmres(b, np.ones(dim))
    print("x=", x_sol_gmres)
    print("res_arr_gm=", res_arr_gm)
    print("Ax=", A.dot(x_sol_gmres))
    
    print("\nSCIPY BiCGSTAB")
    x_sol_bicgstab, exitCode = sp.linalg.bicgstab(A, b, np.ones(dim))
    print("exitcode=", exitCode)
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))
    
    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(A)
    x_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(b, np.ones(dim))
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))

def main(dim):
    print("--== BEGINNING TESTING ==--\n")
    identity_matrix_test(dim)
    symmetric_positive_definite_test(dim)
    symmetric_indefinite_test(dim)
    nonsymmetric_indefinite_test(dim)

if __name__ == "__main__":
    main(8)
    
# do diagonal preconditioners. incomplete LU
# should not take more than one iteration for GMRES to converge with Identity matrix