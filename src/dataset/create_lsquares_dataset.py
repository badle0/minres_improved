import os
import sys
import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
import time
import argparse

# Setting up directory paths for importing custom modules
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, dir_path + '/../lib/')

import minres as mr
import helper_functions as hf

# %% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of the training matrix", default=64)
parser.add_argument("-m", "--number_of_base_ritz_vectors", type=int,
                    help="number of ritz vectors to be used as the base for the dataset", default=10000)
parser.add_argument("--sample_size", type=int,
                    help="number of vectors to be created for dataset. I.e., size of the dataset", default=20000)
parser.add_argument("--theta", type=int,
                    help="see paper for the definition.", default=500)
parser.add_argument("--small_matmul_size", type=int,
                    help="Number of vectors in efficient matrix multiplication", default=200)
parser.add_argument("--dataset_dir", type=str,
                    help="path to the folder containing training matrix")
parser.add_argument("--output_dir", type=str,
                    help="path to the folder the training dataset to be saved")
args = parser.parse_args()

# %% Convert parsed arguments to variables
N = args.resolution
num_ritz_vectors = args.number_of_base_ritz_vectors
small_matmul_size = args.small_matmul_size

# Ensure the output directory exists
import pathlib

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# %% Load the matrix A
if N == 64:
    A_file_name = args.dataset_dir + "/tridiagA_3d" + ".bin"
elif N == 128:
    A_file_name = args.dataset_dir + "/tridiagA_3d" + ".bin"
print("Start loading matrix A from " + A_file_name)
start_matrix = time.time()
A = hf.readA_sparse_from_bin(131072, A_file_name, 'f')
print(f"Shape of matrix A: {A.shape}, dtype: {A.dtype}")
end_matrix = time.time()
print(f"Matrix loading finished in {end_matrix - start_matrix:.2f} seconds.")
# Initialize MINRES solver with the matrix A
MR = mr.MINRESSparse(A)

# %% Generate a random vector for Lanczos iteration
rand_vec_x = np.random.normal(0, 1, [131072])
rand_vec = A.dot(rand_vec_x)

# %% Perform Lanczos Iteration to get the basis vectors (Ritz vectors)
print("Lanczos Iteration is running...")
start_lanczos = time.time()
W, diagonal, sub_diagonal = MR.lanczos_iteration_with_normalization_correction(rand_vec, num_ritz_vectors)
end_lanczos = time.time()
print(f"Lanczos Iteration finished in {end_lanczos - start_lanczos:.2f} seconds.")
print("Lanczos Iteration finished.")

# %% Create the tridiagonal matrix from diagonal and subdiagonal entries
tri_diag = np.zeros([num_ritz_vectors, num_ritz_vectors])
for i in range(1, num_ritz_vectors - 1):
    tri_diag[i, i] = diagonal[i]
    tri_diag[i, i + 1] = sub_diagonal[i]
    tri_diag[i, i - 1] = sub_diagonal[i - 1]
tri_diag[0, 0] = diagonal[0]
tri_diag[0, 1] = sub_diagonal[0]
tri_diag[num_ritz_vectors - 1, num_ritz_vectors - 1] = diagonal[num_ritz_vectors - 1]
tri_diag[num_ritz_vectors - 1, num_ritz_vectors - 2] = sub_diagonal[num_ritz_vectors - 2]

# %% Calculate eigenvalues and eigenvectors of the tridiagonal matrix
print("Calculating eigenvectors of the tridiagonal matrix")
start_eigen = time.time()
ritz_vals, Q0 = np.linalg.eigh(tri_diag)
end_eigen = time.time()
print(f"Eigenvalue calculation finished in {end_eigen - start_eigen:.2f} seconds.")
ritz_vals = np.real(ritz_vals)
Q0 = np.real(Q0)

# %% Transform back to the full space using Ritz vectors
ritz_vectors = np.matmul(W.transpose(), Q0).transpose()

# %% For fast matrix multiplication using Numba
from numba import njit, prange
@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i, j] += A[i, k] * B[k, j]
    return res


# %% Generate the dataset
# lstsquareproblem: normalized A and b
# lstsquareproblem1: normalized bh
# lstsquareproblem10: tridiag A with value from -10 to 10
# lstsquareproblem100: tridiag A with value from -100 to 100

for_outside = int(args.sample_size / small_matmul_size)
b_rhs_temp = np.zeros([small_matmul_size, 131072])
d_rhs_temp = np.zeros([small_matmul_size, N ** 3])
cut_idx = int(num_ritz_vectors / 2) + args.theta

print("Creating Dataset ")
for it in range(0, for_outside):
    t0 = time.time()
    sample_size = small_matmul_size

    # Step 1: Generate diverse x using Ritz vectors
    coef_matrix = np.random.normal(0, 1, [num_ritz_vectors, sample_size])
    coef_matrix[0:cut_idx] = 9 * np.random.normal(0, 1, [cut_idx, sample_size])  # Weight the first half heavily

    # Step 2: Form x using the Ritz vectors
    x_temp = mat_mult(ritz_vectors.transpose(), coef_matrix).transpose()

    # Step 3: Normalize x
    x_temp = x_temp / np.linalg.norm(x_temp, axis=1, keepdims=True)

    # Step 4: Compute b = A * x
    for i in range(small_matmul_size):
        b_rhs_temp[i] = A.dot(x_temp[i])

    # Step 5: Construct the augmented right-hand side vector d = [b; 0]
    for i in range(small_matmul_size):
        d_rhs_temp[i] = np.concatenate([b_rhs_temp[i], np.zeros(131072)])  # d = [b; 0]

    # Step 6: Save the augmented vector d
    for i in range(small_matmul_size):
        with open(args.output_dir + '/d_' + str(it * small_matmul_size + i) + '.npy', 'wb') as f:
            np.save(f, np.array(d_rhs_temp[i], dtype=np.float32))

    t1 = time.time()
    print(f"Batch {it + 1}/{for_outside} finished in {t1 - t0:.2f} seconds.")

print("Dataset creation finished.")
