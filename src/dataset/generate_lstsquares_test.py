import numpy as np
import scipy.sparse as sp
import struct
import os
from scipy.sparse.linalg import eigsh

###############################
# Loading / Saving Functions
###############################
def save_matrix_as_bin(A, filename, dtype='f'):
    A_csr = A.tocsr()
    data = A_csr.data.astype('float32' if dtype == 'f' else 'float64')
    with open(filename, 'wb') as f:
        f.write(struct.pack('<i', A_csr.shape[0]))  # num_rows
        f.write(struct.pack('<i', A_csr.shape[1]))  # num_cols
        f.write(struct.pack('<i', A_csr.nnz))       # nnz
        f.write(struct.pack('<i', len(A_csr.indptr)))
        f.write(struct.pack('<i', A_csr.nnz))
        f.write(data.tobytes())
        f.write(A_csr.indptr.astype('int32').tobytes())
        f.write(A_csr.indices.astype('int32').tobytes())
    print(f"Matrix saved to {filename} with shape {A_csr.shape} and nnz: {A_csr.nnz}")

def save_vector_as_bin(vec, filename, dtype='f'):
    v = vec.astype('float32' if dtype == 'f' else 'float64')
    with open(filename, 'wb') as f:
        f.write(v.tobytes())
    print(f"Vector saved to {filename} with length {len(v)}")

def readA_sparse_from_bin(dim, filenameA, dtype='d'):
    """
    Reads a sparse matrix from a custom binary file.

    Parameters:
        dim (int): The dimension of the matrix (for shape checks if needed).
        filenameA (str): The filename of the binary file.
        dtype (str): The data type ('d' for double, 'f' for float).

    Returns:
        scipy.sparse.csr_matrix: The loaded sparse matrix.
    """
    if dtype == 'd':
        len_data = 8
        data_type = 'float64'
    elif dtype == 'f':
        len_data = 4
        data_type = 'float32'

    with open(filenameA, 'rb') as f:
        # Read header information
        num_rows = struct.unpack('<i', f.read(4))[0]
        num_cols = struct.unpack('<i', f.read(4))[0]
        nnz = struct.unpack('<i', f.read(4))[0]
        outS = struct.unpack('<i', f.read(4))[0]
        innS = struct.unpack('<i', f.read(4))[0]

        # Read data arrays
        data = np.frombuffer(f.read(nnz * len_data), dtype=data_type)
        indptr = np.frombuffer(f.read(outS * 4), dtype='int32')
        indices = np.frombuffer(f.read(nnz * 4), dtype='int32')

    return sp.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))


###############################
# Augmented Matrix Construction
###############################
def spaugment(A, c):
    """
    Constructs a sparse augmented matrix S from matrix A and scalar c.
    Size of A is (n x n). Then S is (2n x 2n).

    S = [[cI, A],
         [A^T, 0]]
    """
    n, m = A.shape
    I = sp.identity(n, format='csr')
    Z = sp.csr_matrix((n, n))

    top = sp.hstack([c * I, A])
    bottom = sp.hstack([A.transpose(), Z])
    S = sp.vstack([top, bottom])
    return S


########################################
# Randomly modify a few entries in A
########################################
def randomly_modify_entries(A, num_changes=3, seed=9999):
    """
    Randomly modifies 'num_changes' existing entries in A's non-zero structure.
    For each selected entry, replace it with a new random value in [0, 1].

    Parameters
    ----------
    A : scipy.sparse.spmatrix (CSR preferred)
        Original matrix. We'll convert to COO for easy index manipulation.
    num_changes : int
        Number of nonzero entries to replace.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    A_modified : scipy.sparse.csr_matrix
        The new matrix with a few random entries changed.
    """
    np.random.seed(seed)

    # Convert to COO to easily manipulate
    A_coo = A.tocoo(copy=True)

    data = A_coo.data
    row = A_coo.row
    col = A_coo.col

    nnz = len(data)
    if num_changes > nnz:
        num_changes = nnz  # can't modify more than we have

    chosen_idxs = np.random.choice(nnz, size=num_changes, replace=False)
    for idx in chosen_idxs:
        data[idx] = np.random.rand()  # random value in [0,1]

    A_modified_coo = sp.coo_matrix((data, (row, col)), shape=A.shape)
    A_modified = A_modified_coo.tocsr()
    return A_modified


###############################
# Main Demonstration
###############################
if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # 1. Specify where your A and S are stored (already generated)
    # ------------------------------------------------------------------------
    output_dir = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquareproblem/test_dataset"
    os.makedirs(output_dir, exist_ok=True)

    A_filename = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquareproblem/tridiagA_3d.bin"
    S_filename = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquareproblem/matrixS_3d64.bin"

    # ------------------------------------------------------------------------
    # 2. Load the matrix A (dim = 131072 for your system) and S from disk
    # ------------------------------------------------------------------------
    print("Loading matrix A from:", A_filename)
    A_original = readA_sparse_from_bin(131072, A_filename, 'f')
    n = A_original.shape[0]
    print(f"Loaded tridiagonal A with shape: {A_original.shape}")

    print("Loading matrix S from:", S_filename)
    S_original = readA_sparse_from_bin(262144, S_filename, 'f')
    print(f"Loaded augmented S with shape: {S_original.shape}")

    # ------------------------------------------------------------------------
    # 3. Create a random x, compute b = A*x for the first test set (Test Set 1)
    # ------------------------------------------------------------------------
    np.random.seed(2024)
    x1 = np.random.rand(n).astype(np.float32)
    b1 = A_original @ x1

    # d1 = [b1; 0]
    d1 = np.concatenate([b1.astype(np.float32), np.zeros(n, dtype=np.float32)])

    # Save A_original, S_original, x1, d1 as Test Set 1
    A1_filename = os.path.join(output_dir, "A_test1.bin")
    S1_filename = os.path.join(output_dir, "S_test1.bin")
    x1_filename = os.path.join(output_dir, "x_test1.bin")
    d1_filename = os.path.join(output_dir, "d_test1.bin")

    save_matrix_as_bin(A_original, A1_filename, dtype='f')
    save_matrix_as_bin(S_original, S1_filename, dtype='f')
    save_vector_as_bin(x1, x1_filename, dtype='f')
    save_vector_as_bin(d1, d1_filename, dtype='f')
    print(f"Test Set 1 saved: {A1_filename}, {S1_filename}, {x1_filename}, {d1_filename}")

    # ------------------------------------------------------------------------
    # 4. Generate MULTIPLE test sets by randomly modifying A.
    #
    #    For each test set:
    #     - Randomly modify A
    #     - Generate a new x
    #     - b = A_mod * x
    #     - S_mod = spaugment(A_mod, c=1.0)
    #     - d_mod = [b; 0]
    #     - Save them
    # ------------------------------------------------------------------------
    num_random_test_sets = 5   # Number of additional test sets you want
    base_seed_for_mods = 9999  # You can adjust or increment for each test

    for i in range(num_random_test_sets):
        test_id = i + 2  # Because we already have 'Test Set 1'

        # Randomly modify A
        A_mod = randomly_modify_entries(
            A_original,
            num_changes=3,
            seed=base_seed_for_mods + i  # vary seed per test set
        )

        # Build x, b, S, d
        np.random.seed(8888 + i)  # vary this too, so x differs each time
        x_mod = np.random.rand(n).astype(np.float32)
        b_mod = A_mod @ x_mod

        S_mod = spaugment(A_mod, c=1.0)
        d_mod = np.concatenate([b_mod.astype(np.float32), np.zeros(n, dtype=np.float32)])

        # Filenames: A_test2.bin, x_test2.bin, etc. (or test3, test4,...)
        A_mod_filename = os.path.join(output_dir, f"A_test{test_id}.bin")
        S_mod_filename = os.path.join(output_dir, f"S_test{test_id}.bin")
        x_mod_filename = os.path.join(output_dir, f"x_test{test_id}.bin")
        d_mod_filename = os.path.join(output_dir, f"d_test{test_id}.bin")

        # Save everything
        save_matrix_as_bin(A_mod, A_mod_filename, dtype='f')
        save_matrix_as_bin(S_mod, S_mod_filename, dtype='f')
        save_vector_as_bin(x_mod, x_mod_filename, dtype='f')
        save_vector_as_bin(d_mod, d_mod_filename, dtype='f')

        print(f"Test Set {test_id} saved: {A_mod_filename}, {S_mod_filename}, "
              f"{x_mod_filename}, {d_mod_filename}")

    print("\n=== All test data generation complete! ===")
