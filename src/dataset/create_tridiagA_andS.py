from scipy.sparse.linalg import eigsh
import numpy as np
import scipy.sparse as sp
import struct
import os


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


def save_matrix_as_bin(A, filename, dtype='d'):
    """Save a sparse matrix A as a custom binary file."""
    A_csr = A.tocsr()  # Ensure the matrix is in CSR format
    data = A_csr.data.astype('float32' if dtype == 'f' else 'float64')

    with open(filename, 'wb') as f:
        # Write the header information
        f.write(struct.pack('<i', A_csr.shape[0]))  # num_rows
        f.write(struct.pack('<i', A_csr.shape[1]))  # num_cols
        f.write(struct.pack('<i', A_csr.nnz))  # nnz
        f.write(struct.pack('<i', len(A_csr.indptr)))  # outS
        f.write(struct.pack('<i', A_csr.nnz))  # innS

        # Write the data arrays
        f.write(data.tobytes())
        f.write(A_csr.indptr.astype('int32').tobytes())
        f.write(A_csr.indices.astype('int32').tobytes())

    print(f"Matrix saved to {filename} with shape {A_csr.shape} and nnz: {A_csr.nnz}")


def readA_sparse_from_bin(dim, filenameA, dtype='d'):
    """
    Reads a sparse matrix from a custom binary file.

    Parameters:
        dim (int): The dimension of the matrix.
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


# Parameters
n = 131072  # Size for A to ensure S is 262,144 x 262,144
c = 1  # Scalar for the identity matrix
directory_path = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquares_problem"
os.makedirs(directory_path, exist_ok=True)

# Generate matrix A
A = random_tridiagonal_matrix(n)

# Check and print the dtype and shape for A
print(f"Matrix A dtype: {A.dtype}, shape: {A.shape}")

# Save matrix A to a binary file
save_matrix_as_bin(A, os.path.join(directory_path, "tridiagA_3d.bin"), dtype='f')
print(f"3D Tridiagonal matrix saved to {os.path.join(directory_path, 'tridiagA_3d.bin')}")

# Load matrix A from the binary file
A_loaded = readA_sparse_from_bin(n, os.path.join(directory_path, "tridiagA_3d.bin"), 'f')
print(f"Loaded matrix A with shape: {A_loaded.shape}, dtype: {A_loaded.dtype}")
print(f"Non-zero elements in loaded A: {A_loaded.nnz}, matching original: {A.nnz == A_loaded.nnz}")

# Construct matrix S
S = spaugment(A, c)

# Check and print the dtype and shape for S
print(f"Matrix S dtype: {S.dtype}, shape: {S.shape}")

# Save matrix S to a binary file
save_matrix_as_bin(S, os.path.join(directory_path, "matrixS_3d64.bin"), dtype='f')
print(f"3D Augmented matrix saved to {os.path.join(directory_path, 'matrixS_3d64.bin')}")

# Load matrix S from the binary file
S_loaded = readA_sparse_from_bin(S.shape[0], os.path.join(directory_path, "matrixS_3d64.bin"), 'f')
print(f"Loaded matrix S with shape: {S_loaded.shape}, dtype: {S_loaded.dtype}")
print(f"Non-zero elements in loaded S: {S_loaded.nnz}, matching original: {S.nnz == S_loaded.nnz}")
