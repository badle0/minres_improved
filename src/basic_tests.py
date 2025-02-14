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
    n, m = A.shape
    I = sp.identity(n, format='csr')
    Z = sp.csr_matrix((n, n))
    top = sp.hstack([c * I, A])
    bottom = sp.hstack([A.transpose(), Z])
    S = sp.vstack([top, bottom])
    return S

# Generate a sparse, indefinite, nonsymmetric matrix(size, seed=42)
def generate_indefinite_nonsymmetric_matrix(size, density=0.05, seed=42):
    np.random.seed(seed)
    
    # Step 1: Generate a sparse random matrix (nonsymmetric)
    A = sp.random(size, size, density=density, format='csr', data_rvs=np.random.rand)
    
    # Step 2: Generate a random skew-symmetric matrix by creating a matrix and subtracting its transpose
    skew_symmetric = sp.random(size, size, density=density, format='csr', data_rvs=np.random.rand)
    skew_symmetric = skew_symmetric - skew_symmetric.T  # make it skew-symmetric
    
    # Step 3: Combine the nonsymmetric matrix and skew-symmetric part
    A = A + skew_symmetric  # Adding skew-symmetric matrix makes it nonsymmetric
    
    # Step 4: Convert to LIL format for efficient diagonal modification
    A = A.tolil()  # Convert to LIL format
    
    # Step 5: Add negative diagonal entries to ensure indefiniteness
    diag_vals = np.random.uniform(-1, 1, size)
    for i in range(size):
        A[i, i] = diag_vals[i]  # Set diagonal values directly
    
    # Step 6: Convert back to CSR format for efficient arithmetic
    A = A.tocsr()
    
    return A


def identity_matrix_test(dim):
    print("IDENTITY MATRIX")
    A = np.eye(dim)
    b = np.arange(dim)
    print("A=\n", A)
    print("Condition number of A:", np.linalg.cond(A))
    print("b=", b)

    print("\nSCIPY MINRES")
    start_time = time.time()
    x_sol_mres, exitCode = sp.linalg.minres(A, b,show=True)
    end_time = time.time()
    print('exitcode=', exitCode)
    print("x=", x_sol_mres)
    print("Time taken:", end_time - start_time, "seconds")
    
    print("\nCUSTOM MINRES")
    MR = MINRESSparse(A)
    start_time = time.time()
    x_sol_mr = MR.minres(b, np.ones(b.shape))
    end_time = time.time()
    print("x_sol_mr=", x_sol_mr)
    print("Time taken:", end_time - start_time, "seconds")

    print("\nSCIPY GMRES")
    start_time = time.time()
    x_sol_gmres, exitCode = sp.linalg.gmres(A, b)
    end_time = time.time()
    print("exitcode=", exitCode)
    print("x_sol_gmres=", x_sol_gmres)
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    start_time = time.time()
    x_sol_gm, res_arr_gm = GM.gmres(b, np.ones(dim))
    end_time = time.time()
    print("x_sol_gm=", x_sol_gm)
    print("res_arr_gm=", res_arr_gm)
    print("Time taken:", end_time - start_time, "seconds")

    print("\nSCIPY BiCGSTAB")
    start_time = time.time()
    x_sol_bicgstab, exitCode = sp.linalg.bicgstab(A, b)
    end_time = time.time()
    print("exitcode=", exitCode)
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("Time taken:", end_time - start_time, "seconds")
    
    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(A)
    start_time = time.time()
    x_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(b, np.ones(dim))
    end_time = time.time()
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Time taken:", end_time - start_time, "seconds\n")


def symmetric_positive_definite_test(dim):
    print("\nSYMMETRIC POSITIVE DEFINITE MATRIX")
    np.random.seed(42)
    A = np.random.rand(dim, dim)
    A = np.dot(A, A.T) + dim * np.eye(dim)
    b = np.arange(dim)
    print("A=\n", pd.DataFrame(A))
    print("Condition number of A:", np.linalg.cond(A))
    print("b=", b)

    print("\nSCIPY MINRES")
    start_time = time.time()
    x_sol_mres, exitCode = sp.linalg.minres(A, b, show=True)
    end_time = time.time()
    print('exitcode=', exitCode)
    print("x=", x_sol_mres)
    print("Ax=", A.dot(x_sol_mres))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM MINRES")
    MR = MINRESSparse(A)
    start_time = time.time()
    x_sol_mr = MR.minres(b, np.ones(b.shape))
    end_time = time.time()
    print("x_sol_mr=", x_sol_mr)
    print("Ax=", A.dot(x_sol_mr))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nSCIPY GMRES")
    start_time = time.time()
    x_sol_gmres, exitCode = sp.linalg.gmres(A, b)
    end_time = time.time()
    print('exitcode=', exitCode)
    print("x=", x_sol_gmres)
    print("Ax=", A.dot(x_sol_gmres))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    start_time = time.time()
    x_sol_gm, res_arr_gm = GM.gmres(b, np.ones(dim))
    end_time = time.time()
    print("x=", x_sol_gm)
    print("res_arr_gm=", res_arr_gm)
    print("Ax=", A.dot(x_sol_gm))
    print("Time taken:", end_time - start_time, "seconds")
    
    print("\nSCIPY BiCGSTAB")
    start_time = time.time()
    x_sol_bicgstab, exitCode = sp.linalg.bicgstab(A, b)
    end_time = time.time()
    print("exitcode=", exitCode)
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(A)
    start_time = time.time()
    x_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(b, np.ones(dim))
    end_time = time.time()
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))
    print("Time taken:", end_time - start_time, "seconds")


def symmetric_indefinite_test(dim):
    print("\nSYMMETRIC INDEFINITE")
    tri_A = random_tridiagonal_matrix(dim)
    b = np.arange(dim)
    S = spaugment(tri_A, 1)
    d = np.append(b, np.zeros(dim))
    print("SOLVING Sy=d for spaugmented system...")
    print("S=\n", pd.DataFrame(S.toarray()))
    print("Condition number of S:", np.linalg.cond(S.toarray()))
    print("d=", d)

    print("\nSCIPY MINRES")
    start_time = time.time()
    y_sol_mres, exitCode = sp.linalg.minres(S, d, show=True)
    end_time = time.time()
    print('exitcode=', exitCode)
    print("y=", y_sol_mres)
    print("Sy=", S.dot(y_sol_mres))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM MINRES")
    MR = MINRESSparse(S)
    start_time = time.time()
    y_sol_mr = MR.minres(d, np.ones(d.shape))
    end_time = time.time()
    print("y_sol_mr=", y_sol_mr)
    print("Sy=", S.dot(y_sol_mr))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nSCIPY GMRES")
    start_time = time.time()
    y_sol_gmres, exitCode = sp.linalg.gmres(S, d)
    end_time = time.time()
    print('exitcode=', exitCode)
    print("y=", y_sol_gmres)
    print("Sy=", S.dot(y_sol_gmres))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(S)
    start_time = time.time()
    y_sol_gmres, res_arr_gm = GM.gmres(d, np.ones(dim * 2))
    end_time = time.time()
    print("y=", y_sol_gmres)
    print("res_arr_gm=", res_arr_gm)
    print("Sy=", S.dot(y_sol_gmres))
    print("Time taken:", end_time - start_time, "seconds")
    
    print("\nSCIPY BiCGSTAB")
    start_time = time.time()
    y_sol_bicgstab, exitCode = sp.linalg.bicgstab(S, d, np.ones(dim * 2))
    end_time = time.time()
    print("exitcode=", exitCode)
    print("y_sol_bicgstab=", y_sol_bicgstab)
    print("Sy=", S.dot(y_sol_bicgstab))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(S)
    start_time = time.time()
    y_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(d, np.ones(dim * 2))
    end_time = time.time()
    print("y_sol_bicgstab=", y_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Sy=", S.dot(y_sol_bicgstab))
    print("Time taken:", end_time - start_time, "seconds")


def nonsymmetric_indefinite_test(dim):
    print("\nNONSYMMETRIC INDEFINITE MATRIX")
    A = generate_indefinite_nonsymmetric_matrix(dim)
    b = np.arange(dim)
    print("A=\n", pd.DataFrame(A.toarray()))
    print("Condition number of A:", np.linalg.cond(A.toarray()))
    print("b=", b)

    print("\nSCIPY GMRES")
    start_time = time.time()
    x_sol_gmres, exitCode = sp.linalg.gmres(A, b)
    end_time = time.time()
    print('exitcode=', exitCode)
    print("x=", x_sol_gmres)
    print("Ax=", A.dot(x_sol_gmres))
    print("Time taken:", end_time - start_time, "seconds")

    print("\nCUSTOM GMRES")
    GM = GMRESSparse(A)
    start_time = time.time()
    x_sol_gmres, res_arr_gm = GM.gmres(b, np.ones(dim))
    end_time = time.time()
    print("x=", x_sol_gmres)
    print("res_arr_gm=", res_arr_gm)
    print("Ax=", A.dot(x_sol_gmres))
    print("Time taken:", end_time - start_time, "seconds")
    
    print("\nSCIPY BiCGSTAB")
    start_time = time.time()
    x_sol_bicgstab, exitCode = sp.linalg.bicgstab(A, b, np.ones(dim))
    end_time = time.time()
    print("exitcode=", exitCode)
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))
    print("Time taken:", end_time - start_time, "seconds")
    
    print("\nCUSTOM BiCGSTAB")
    bicgstab = BiCGSTABSparse(A)
    start_time = time.time()
    x_sol_bicgstab, res_arr_bicgstab = bicgstab.bicgstab(b, np.ones(dim))
    end_time = time.time()
    print("x_sol_bicgstab=", x_sol_bicgstab)
    print("res_arr_bicgstab=", res_arr_bicgstab)
    print("Ax=", A.dot(x_sol_bicgstab))
    print("Time taken:", end_time - start_time, "seconds")

def main(dim):
    print("--== BEGINNING TESTING ==--\n")
    identity_matrix_test(dim)
    symmetric_positive_definite_test(dim)
    symmetric_indefinite_test(dim)
    nonsymmetric_indefinite_test(dim)

if __name__ == "__main__":
    main(23) # 4 < dim < 24
             # at dim=25, custom bicgstab works, but not scipy.
             # at dim=24, scipy and custom bicgstab work. Both GMRES
             # 
    
# do diagonal preconditioners. incomplete LU
# should not take more than one iteration for GMRES to converge with Identity matrix