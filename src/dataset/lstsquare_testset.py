import numpy as np
import os
import scipy.sparse as sp
import struct
from scipy.sparse.linalg import svds

# Helper function to save matrix as binary file
def save_matrix_as_bin(A, filename, dtype='d'):
    A_csr = A.tocsr()  # Ensure the matrix is in CSR format
    data = A_csr.data.astype('float32' if dtype == 'f' else 'float64')

    with open(filename, 'wb') as f:
        f.write(struct.pack('<i', A_csr.shape[0]))  # num_rows
        f.write(struct.pack('<i', A_csr.shape[1]))  # num_cols
        f.write(struct.pack('<i', A_csr.nnz))  # nnz
        f.write(struct.pack('<i', len(A_csr.indptr)))  # outS
        f.write(struct.pack('<i', A_csr.nnz))  # innS

        f.write(data.tobytes())
        f.write(A_csr.indptr.tobytes())
        f.write(A_csr.indices.tobytes())

    print(f"Matrix saved to {filename} with shape {A_csr.shape} and nnz: {A_csr.nnz}")

# Generate tridiagonal matrix and normalize it
def generate_tridiagonal_matrix(n, seed=42):
    np.random.seed(seed)
    main_diag = np.random.rand(n)
    off_diag = np.random.rand(n - 1)
    A = sp.diags(diagonals=[off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format='csr')
    u, s, vt = svds(A, k=1)  # Get the largest singular value
    A_normalized = A / s[0]  # Normalize A by its largest singular value
    return A_normalized

# Function to perturb matrix A
def perturb_matrix(A, perturbation_factor=0.01, seed=None):
    np.random.seed(seed)
    A = A.tolil()  # Convert to LIL format for efficient modifications
    n = A.shape[0]
    num_perturbations = int(0.05 * n)  # Perturb 5% of the elements
    for _ in range(num_perturbations):
        i = np.random.randint(0, n)
        j = np.random.randint(max(0, i-1), min(n, i+2))  # Perturb diagonal or near diagonal entries
        A[i, j] += perturbation_factor * np.random.randn()
    return A.tocsr()

# Augment matrix A to create matrix S
def spaugment(A, c):
    n, m = A.shape
    I = sp.identity(n)
    Z = sp.csr_matrix((m, m))  # Zero matrix
    top = sp.hstack([c * I, A])
    bottom = sp.hstack([A.transpose(), Z])
    S = sp.vstack([top, bottom])
    return S

# Set parameters
n = 131072  # Matrix size
c = 1
num_test_vectors = 10  # Number of test vectors
output_dir = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquareproblem/test_dataset"
os.makedirs(output_dir, exist_ok=True)

# **Test Set 1**: Matrix A is constant
A_constant = generate_tridiagonal_matrix(n, seed=100)
S_constant = spaugment(A_constant, c)

# Generate test vectors for Test Set 1
for idx in range(num_test_vectors):
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)  # Normalize x
    b = A_constant.dot(x)
    d = np.concatenate([b, np.zeros(n)]).astype(np.float32)

    # Save matrix S and vector d
    matrix_file = os.path.join(output_dir, f"matrixS_constant_{idx}.bin")
    vector_file = os.path.join(output_dir, f"rhs_vector_constant_{idx}.bin")
    save_matrix_as_bin(S_constant, matrix_file, dtype='f')
    d.tofile(vector_file)
    print(f"Test Set 1: Saved constant matrix and vector for test case {idx + 1}")

# **Test Set 2**: Matrix A is perturbed randomly for each test
for idx in range(num_test_vectors):
    A_perturbed = perturb_matrix(A_constant, perturbation_factor=0.01, seed=101 + idx)
    S_perturbed = spaugment(A_perturbed, c)

    x = np.random.rand(n)
    x = x / np.linalg.norm(x)  # Normalize x
    b = A_perturbed.dot(x)
    d = np.concatenate([b, np.zeros(n)]).astype(np.float32)

    # Save matrix S and vector d
    matrix_file = os.path.join(output_dir, f"matrixS_perturbed_{idx}.bin")
    vector_file = os.path.join(output_dir, f"rhs_vector_perturbed_{idx}.bin")
    save_matrix_as_bin(S_perturbed, matrix_file, dtype='f')
    d.tofile(vector_file)
    print(f"Test Set 2: Saved perturbed matrix and vector for test case {idx + 1}")

print("Test dataset generation completed.")
