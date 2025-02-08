import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg._isolve import minres
import struct
import sys
import os
import tensorflow as tf

# Import custom modules
# Setting up directory paths for importing custom modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../lib'))
sys.path.insert(1, lib_path)
import minres as mr
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

    return S

n = 8  # Size for A to ensure S is 262,144 x 262,144
c = 1

A = random_tridiagonal_matrix(n)
print(f"Matrix A dtype: {A.dtype}, shape: {A.shape}")
S = spaugment(A, c)
print(f"Matrix S dtype: {S.dtype}, shape: {S.shape}")
b = np.arange(n, dtype=A.dtype)
print("b =", b)
d = np.concatenate([b, np.zeros(n, dtype=A.dtype)])
print("d =", d)
print("\nScipy MINRES on Symmetric Indefinite System")
x_sol_minres, exitCode = minres(S,d,show=True)
print('exit code:', exitCode)
print('x=', x_sol_minres)
print("Ax=", S.dot(x_sol_minres))
print("b=", b)

print("\nCustom MINRES")
MR=mr.MINRESSparse(S)
x_sol_mr = MR.minres(d, np.zeros(d.shape))
print("x=", x_sol_mr)
print("Ax=", S.dot(x_sol_mr))
print("b=", b)