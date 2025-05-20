import os, sys, time, argparse, pathlib
import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
from numba import njit, prange

# ---------------------------------------------------------------------
#  paths
# ---------------------------------------------------------------------
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, os.path.join(dir_path, "../../lib/"))

import helper_functions as hf
import conjugate_residual as cr

# ---------------------------------------------------------------------
# 1. command-line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution",             type=int,
                    choices=[64, 128], default=64)
parser.add_argument("-m", "--number_of_base_ritz_vectors",
                    type=int, default=10000)
parser.add_argument("--sample_size",       type=int,  default=20000)
parser.add_argument("--theta",             type=int,  default=500)
parser.add_argument("--small_matmul_size", type=int,  default=200)
parser.add_argument("--dataset_dir",       type=str,
                    help="folder that *already* contains A_indefN*.bin")
parser.add_argument("--output_dir",        type=str,
                    help="folder to save .npy RHS vectors")
args = parser.parse_args()

# ---------------------------------------------------------------------
# 2. basic params
# ---------------------------------------------------------------------
N                 = args.resolution
num_ritz_vectors  = args.number_of_base_ritz_vectors
small_matmul_size = args.small_matmul_size

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 3. load **indefinite** matrix
# ---------------------------------------------------------------------
if N == 64:
    A_file = os.path.join(args.dataset_dir, "A_indefN64.bin")
elif N == 128:
    A_file = os.path.join(args.dataset_dir, "A_indefN128.bin")
else:
    raise ValueError("Unsupported resolution")

A = hf.readA_sparse(N, A_file, 'f')               # float32 CSR

# ---------------------------------------------------------------------
# 4. CR-based Lanczos
# ---------------------------------------------------------------------
CR   = cr.ConjugateResidualSparse(A)
r0   = A @ np.random.normal(0, 1, N**3)           # random RHS
print("Lanczos iteration is running ...")
W, diagonal, sub_diag = CR.lanczos_iteration_with_normalization_correction(
                            r0, num_ritz_vectors)
print("Lanczos iteration finished.")

# ---------------------------------------------------------------------
# 5. build tridiagonal T  (unchanged)
# ---------------------------------------------------------------------
T = np.zeros((num_ritz_vectors, num_ritz_vectors), dtype=np.float32)
for i in range(1, num_ritz_vectors-1):
    T[i, i]     = diagonal[i]
    T[i, i+1]   = sub_diag[i]
    T[i, i-1]   = sub_diag[i-1]
T[0, 0]                             = diagonal[0]
T[0, 1]                             = sub_diag[0]
T[-1, -1]                           = diagonal[-1]
T[-1, -2]                           = sub_diag[-1]

# ---------------------------------------------------------------------
# 6. Ritz eigen-analysis
# ---------------------------------------------------------------------
print("Computing eigenpairs of tridiagonal matrix ...")
ritz_vals, Q0 = np.linalg.eigh(T)
ritz_vectors  = (W.T @ Q0).T               # shape (m, n)
ritz_vals     = np.real(ritz_vals)

# separate by sign
neg_idx = np.where(ritz_vals < -1e-8)[0]
pos_idx = np.where(ritz_vals >  1e-8)[0]

print(f"   #neg = {len(neg_idx)},  #pos = {len(pos_idx)}")

# ---------------------------------------------------------------------
# 7. parallel mat-mult helper           (unchanged)
# ---------------------------------------------------------------------
@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), dtype=A.dtype)
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i, j] += A[i, k] * B[k, j]
    return res

# ---------------------------------------------------------------------
# 8. sample residuals and save
# ---------------------------------------------------------------------
outer_loops = args.sample_size // small_matmul_size
b_temp      = np.zeros((small_matmul_size, N**3), dtype=np.float32)

print("Creating dataset ...")
for it in range(outer_loops):
    # -----------------------------------------------------------------
    # 8.1 coefficient matrix: heavier weight on negative Ritz vectors
    # -----------------------------------------------------------------
    coef_neg = np.random.normal(0, 3,  (len(neg_idx), small_matmul_size))
    coef_pos = np.random.normal(0, 1,  (len(pos_idx), small_matmul_size))
    coef_mat = np.vstack([coef_neg, coef_pos])

    # -----------------------------------------------------------------
    # 8.2 assemble subset of Ritz vectors used for mat-mult
    # -----------------------------------------------------------------
    ritz_stack = ritz_vectors[np.hstack([neg_idx, pos_idx])]

    # -----------------------------------------------------------------
    # 8.3 batched multiplication & normalisation
    # -----------------------------------------------------------------
    b_temp[:] = mat_mult(ritz_stack.T, coef_mat).T
    b_temp   /= np.linalg.norm(b_temp, axis=1, keepdims=True)

    # -----------------------------------------------------------------
    # 8.4 dump to .npy
    # -----------------------------------------------------------------
    for local_idx in range(small_matmul_size):
        global_idx = it * small_matmul_size + local_idx
        fname = os.path.join(args.output_dir, f"b_{global_idx}.npy")
        np.save(fname, b_temp[local_idx])

    elapsed = time.time() - (time.time() - 0)   # dummy to keep resemblance
    print(f"  batch {it+1}/{outer_loops} written", flush=True)

print("Training dataset created.")
