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
from gcr import GeneralizedConjugateResidual
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


# Different preconditioners
def no_preconditioner(A):
    return sp.linalg.LinearOperator(A.shape, matvec=lambda x: x)

def diagonal_preconditioner(A):
    M_inv = 1.0 / A.diagonal()
    return sp.linalg.LinearOperator(A.shape, matvec=lambda x: M_inv * x)

def ilu_preconditioner(A):
    try:
        ilu = sp.linalg.spilu(A)
        return sp.linalg.LinearOperator(A.shape, matvec=ilu.solve)
    except RuntimeError:
        print("ILU preconditioning failed, using no preconditioner.")
        return no_preconditioner(A)

def print_solver_results(solver_name, solver_func, A, b, preconditioner=None, custom_solver=False, method_name=None):
    print(f"\n{solver_name}")
    start_time = time.time()
    try:
        x0 = np.zeros_like(b, dtype=np.float64)
        b = b.astype(np.float64)
        if custom_solver:
            solver = solver_func(A)
            if preconditioner:
                result = getattr(solver, method_name)(b, x0=x0, M=preconditioner)
            else:
                result = getattr(solver, method_name)(b, x0=x0)
        else:
            if preconditioner:
                result = solver_func(A, b, x0=x0, M=preconditioner)
            else:
                result = solver_func(A, b, x0=x0)
        
        # Check if the result is a tuple (x, exitcode) or just x
        if isinstance(result, tuple):
            x, exitcode = result
        else:
            x = result
            exitcode = None
    except Exception as e:
        print(f"Solver {solver_name} failed with exception: {e}")
        return

    end_time = time.time()
    print("exitcode=", exitcode)
    print("x=", x)
    print("Ax=", A.dot(x))
    print("Time taken:", end_time - start_time, "seconds")

def identity_matrix_test(dim):
    print("IDENTITY MATRIX")
    A = np.eye(dim)
    b = np.arange(dim, dtype=np.float64)
    print("A=\n", A)
    print("Condition number of A:", np.linalg.cond(A))
    print("b=", b)

    # MINRES tests
    print_solver_results("SCIPY MINRES", sp.linalg.minres, A, b)
    print_solver_results("CUSTOM MINRES", MINRESSparse, A, b, custom_solver=True, method_name='minres')
    
    # GMRES tests
    print_solver_results("SCIPY GMRES", sp.linalg.gmres, A, b)
    print_solver_results("CUSTOM GMRES", GMRESSparse, A, b, custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (diagonal preconditioner)", sp.linalg.gmres, A, b, diagonal_preconditioner(A))
    print_solver_results("CUSTOM GMRES (diagonal preconditioner)", GMRESSparse, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (ilu preconditioner)", sp.linalg.gmres, A, b, ilu_preconditioner(A))
    print_solver_results("CUSTOM GMRES (ilu preconditioner)", GMRESSparse, A, b, ilu_preconditioner(A), custom_solver=True, method_name='gmres')
    
    # BiCGSTAB tests
    print_solver_results("SCIPY BiCGSTAB", sp.linalg.bicgstab, A, b)
    print_solver_results("CUSTOM BiCGSTAB", BiCGSTABSparse, A, b, custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (diagonal preconditioner)", sp.linalg.bicgstab, A, b, diagonal_preconditioner(A))
    print_solver_results("CUSTOM BiCGSTAB (diagonal preconditioner)", BiCGSTABSparse, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (ilu preconditioner)", sp.linalg.bicgstab, A, b, ilu_preconditioner(A))
    print_solver_results("CUSTOM BiCGSTAB (ilu preconditioner)", BiCGSTABSparse, A, b, ilu_preconditioner(A), custom_solver=True, method_name='bicgstab')
    
    # GCR tests
    print_solver_results("CUSTOM GCR", GeneralizedConjugateResidual, A, b,custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (diagonal preconditioner)", GeneralizedConjugateResidual, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (ilu preconditioner)", GeneralizedConjugateResidual, A, b, ilu_preconditioner(A), custom_solver=True, method_name='gcr')



def symmetric_positive_definite_test(dim):
    print("\nSYMMETRIC POSITIVE DEFINITE MATRIX")
    np.random.seed(42)
    A = np.random.rand(dim, dim)
    A = np.dot(A, A.T) + dim * np.eye(dim)
    b = np.arange(dim, dtype=np.float64)
    print("A=\n", pd.DataFrame(A))
    print("Condition number of A:", np.linalg.cond(A))
    print("b=", b)

    # MINRES tests
    print_solver_results("SCIPY MINRES", sp.linalg.minres, A, b)
    print_solver_results("CUSTOM MINRES", MINRESSparse, A, b, custom_solver=True, method_name='minres')
    
    # GMRES tests
    print_solver_results("SCIPY GMRES", sp.linalg.gmres, A, b)
    print_solver_results("CUSTOM GMRES", GMRESSparse, A, b, custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (diagonal preconditioner)", sp.linalg.gmres, A, b, diagonal_preconditioner(A))
    print_solver_results("CUSTOM GMRES (diagonal preconditioner)", GMRESSparse, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (ilu preconditioner)", sp.linalg.gmres, A, b, ilu_preconditioner(A))
    print_solver_results("CUSTOM GMRES (ilu preconditioner)", GMRESSparse, A, b, ilu_preconditioner(A), custom_solver=True, method_name='gmres')
    
    # BiCGSTAB tests
    print_solver_results("SCIPY BiCGSTAB", sp.linalg.bicgstab, A, b)
    print_solver_results("CUSTOM BiCGSTAB", BiCGSTABSparse, A, b, custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (diagonal preconditioner)", sp.linalg.bicgstab, A, b, diagonal_preconditioner(A))
    print_solver_results("CUSTOM BiCGSTAB (diagonal preconditioner)", BiCGSTABSparse, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (ilu preconditioner)", sp.linalg.bicgstab, A, b, ilu_preconditioner(A))
    print_solver_results("CUSTOM BiCGSTAB (ilu preconditioner)", BiCGSTABSparse, A, b, ilu_preconditioner(A), custom_solver=True, method_name='bicgstab')

    # GCR tests
    print_solver_results("CUSTOM GCR", GeneralizedConjugateResidual, A, b, custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (diagonal preconditioner)", GeneralizedConjugateResidual, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (ilu preconditioner)", GeneralizedConjugateResidual, A, b, ilu_preconditioner(A), custom_solver=True, method_name='gcr')


def symmetric_indefinite_test(dim):
    print("\nSYMMETRIC INDEFINITE")
    tri_A = random_tridiagonal_matrix(dim)
    b = np.arange(dim, dtype=np.float64)
    S = spaugment(tri_A, 1)
    d = np.append(b, np.zeros(dim))
    print("SOLVING Sy=d for spaugmented system...")
    print("S=\n", pd.DataFrame(S.toarray()))
    print("Condition number of S:", np.linalg.cond(S.toarray()))
    print("d=", d)

    # MINRES tests
    print_solver_results("SCIPY MINRES", sp.linalg.minres, S, d)
    print_solver_results("CUSTOM MINRES", MINRESSparse, S, d, custom_solver=True, method_name='minres')
    
    # GMRES tests
    print_solver_results("SCIPY GMRES", sp.linalg.gmres, S, d)
    print_solver_results("CUSTOM GMRES", GMRESSparse, S, d, custom_solver=True, method_name='gmres')
    # DIAGONAL PRECONDITIONER WILL NOT WORK IF ANY DIAGONAL ENTRY IS ZERO
    #print_solver_results("SCIPY GMRES (diagonal preconditioner)", sp.linalg.gmres, S, d, diagonal_preconditioner(S))
    #print_solver_results("CUSTOM GMRES (diagonal preconditioner)", GMRESSparse, S, d, diagonal_preconditioner(S), custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (ilu preconditioner)", sp.linalg.gmres, S, d, ilu_preconditioner(S))
    print_solver_results("CUSTOM GMRES (ilu preconditioner)", GMRESSparse, S, d, ilu_preconditioner(S), custom_solver=True, method_name='gmres')
    
    # BiCGSTAB tests
    print_solver_results("SCIPY BiCGSTAB", sp.linalg.bicgstab, S, b)
    print_solver_results("CUSTOM BiCGSTAB", BiCGSTABSparse, S, d, custom_solver=True, method_name='bicgstab')
    # DIAGONAL PRECONDITIONER WILL NOT WORK IF ANY DIAGONAL ENTRY IS ZERO
    # print_solver_results("SCIPY BiCGSTAB (diagonal preconditioner)", sp.linalg.bicgstab, S, d, diagonal_preconditioner(S))
    # print_solver_results("CUSTOM BiCGSTAB (diagonal preconditioner)", BiCGSTABSparse, S, d, diagonal_preconditioner(S), custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (ilu preconditioner)", sp.linalg.bicgstab, S, d, ilu_preconditioner(S))
    print_solver_results("CUSTOM BiCGSTAB (ilu preconditioner)", BiCGSTABSparse, S, d, ilu_preconditioner(S), custom_solver=True, method_name='bicgstab')

    # GCR tests
    print_solver_results("CUSTOM GCR", GeneralizedConjugateResidual, S, d, custom_solver=True, method_name='gcr')
    #print_solver_results("CUSTOM GCR (diagonal preconditioner)", GeneralizedConjugateResidual, S, d, diagonal_preconditioner(S), custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (ilu preconditioner)", GeneralizedConjugateResidual, S, d, ilu_preconditioner(S), custom_solver=True, method_name='gcr')

def nonsymmetric_indefinite_test(dim):
    print("\nNONSYMMETRIC INDEFINITE MATRIX")
    A = generate_indefinite_nonsymmetric_matrix(dim)
    b = np.arange(dim, dtype=np.float64)
    print("A=\n", pd.DataFrame(A.toarray()))
    print("Condition number of A:", np.linalg.cond(A.toarray()))
    print("b=", b)

    # GMRES tests
    print_solver_results("SCIPY GMRES", sp.linalg.gmres, A, b)
    print_solver_results("CUSTOM GMRES", GMRESSparse, A, b, custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (diagonal preconditioner)", sp.linalg.gmres, A, b, diagonal_preconditioner(A))
    print_solver_results("CUSTOM GMRES (diagonal preconditioner)", GMRESSparse, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='gmres')
    print_solver_results("SCIPY GMRES (ilu preconditioner)", sp.linalg.gmres, A, b, ilu_preconditioner(A))
    print_solver_results("CUSTOM GMRES (ilu preconditioner)", GMRESSparse, A, b, ilu_preconditioner(A), custom_solver=True, method_name='gmres')
    
    # BiCGSTAB tests
    print_solver_results("SCIPY BiCGSTAB", sp.linalg.bicgstab, A, b)
    print_solver_results("CUSTOM BiCGSTAB", BiCGSTABSparse, A, b, custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (diagonal preconditioner)", sp.linalg.bicgstab, A, b, diagonal_preconditioner(A))
    print_solver_results("CUSTOM BiCGSTAB (diagonal preconditioner)", BiCGSTABSparse, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='bicgstab')
    print_solver_results("SCIPY BiCGSTAB (ilu preconditioner)", sp.linalg.bicgstab, A, b, ilu_preconditioner(A))
    print_solver_results("CUSTOM BiCGSTAB (ilu preconditioner)", BiCGSTABSparse, A, b, ilu_preconditioner(A), custom_solver=True, method_name='bicgstab')

    # GCR tests
    print_solver_results("CUSTOM GCR", GeneralizedConjugateResidual, A, b, custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (diagonal preconditioner)", GeneralizedConjugateResidual, A, b, diagonal_preconditioner(A), custom_solver=True, method_name='gcr')
    print_solver_results("CUSTOM GCR (ilu preconditioner)", GeneralizedConjugateResidual, A, b, ilu_preconditioner(A), custom_solver=True, method_name='gcr')

def main(dim):
    print("--== BEGINNING TESTING ==--\n")
    #identity_matrix_test(dim)
    #symmetric_positive_definite_test(dim)
    #symmetric_indefinite_test(dim)
    nonsymmetric_indefinite_test(dim)

if __name__ == "__main__":
    main(128) # 4 < dim < 24
             # at dim=25, custom bicgstab works, but not scipy.
             # at dim=24, scipy and custom bicgstab work. Both GMRES fail.