import os
import sys
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import numpy as np

# Setting up directory paths for importing custom modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../../lib'))
sys.path.insert(1, lib_path)

import helper_functions as hf

# Number of test matrices
num_tests = 6

N = 64

# Directory containing your S_test#.bin files
test_dataset_dir = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquareproblem/test_dataset"
output_dir = os.path.join(test_dataset_dir, "iLU")

os.makedirs(output_dir, exist_ok=True)

for i in range(num_tests):
    test_id = i + 1

    # Build the filename for each S_test#.bin
    A_file_name = os.path.join(test_dataset_dir, f"S_test{test_id}.bin")

    # Read the matrix from disk (dimension = 64 if that's what your file contains)
    print(f"\nReading augmented matrix from: {A_file_name}")
    A = hf.readA_sparse(N, A_file_name, 'f')
    print(f"S_test{test_id}.bin shape: {A.shape}")

    # Generate the incomplete LU preconditioner
    A_csr = A.tocsr()
    ilu = spla.spilu(A_csr, drop_tol=1e-6)

    L = ilu.L
    U = ilu.U
    print(f"Computed ILU for S_test{test_id}.bin. Saving L and U...")

    # Build output filenames, e.g. L1.npz, U1.npz, then L2.npz, U2.npz, ...
    L_filename = os.path.join(output_dir, f"L{test_id}.npz")
    U_filename = os.path.join(output_dir, f"U{test_id}.npz")

    # Save L and U components
    sparse.save_npz(L_filename, L)
    sparse.save_npz(U_filename, U)

    print(f"Saved:\n  {L_filename}\n  {U_filename}")

print("\n=== Done generating all ILU factors for the 6 test matrices! ===")
