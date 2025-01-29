import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import os
import struct

def construct_3d_poisson_matrix(N):
    """Generate a standard 3D Poisson matrix using finite differences on an N x N x N grid with float32 precision."""
    # Use float32 data type for identity matrices
    I = sp.eye(N, dtype=np.float32, format='csr')
    # Use float32 data type for ones vector
    e = np.ones(N, dtype=np.float32)
    # Create the tridiagonal matrix T with float32 data type
    T = sp.diags([-e, 2 * e, -e], [-1, 0, 1], shape=(N, N), dtype=np.float32, format='csr')
    # Compute the Kronecker products and sum them to get A
    A = (sp.kron(sp.kron(I, I), T, format='csr') +
         sp.kron(sp.kron(I, T), I, format='csr') +
         sp.kron(sp.kron(T, I), I, format='csr'))
    return A

def verify_poisson_matrix(A):
    # Check if the matrix is symmetric
    is_symmetric = (A - A.T).nnz == 0
    print(f"Symmetric: {is_symmetric}")

    # Check if the matrix is positive definite
    try:
        # Positive definite if all eigenvalues are positive
        eigenvalues = spla.eigsh(A, k=6, which='SM', return_eigenvectors=False)
        is_positive_definite = np.all(eigenvalues > 0)
    except:
        is_positive_definite = False
    print(f"Positive Definite: {is_positive_definite}")


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


def readA_sparse(dim, filenameA, dtype='d'):
    """Read a sparse matrix from a custom binary file."""
    if dtype == 'd':
        len_data = 8  # 8 bytes for float64
        data_format = 'd'
    elif dtype == 'f':
        len_data = 4  # 4 bytes for float32
        data_format = 'f'
    else:
        raise ValueError("Unsupported data type.")

    with open(filenameA, 'rb') as f:
        length = 4
        b = f.read(length)
        num_rows = struct.unpack('i', b)[0]
        b = f.read(length)
        num_cols = struct.unpack('i', b)[0]
        b = f.read(length)
        nnz = struct.unpack('i', b)[0]
        b = f.read(length)
        outS = struct.unpack('i', b)[0]
        b = f.read(length)
        innS = struct.unpack('i', b)[0]

        data = np.frombuffer(f.read(nnz * len_data), dtype=np.float32 if dtype == 'f' else np.float64)
        indptr = np.frombuffer(f.read(outS * 4), dtype=np.int32)
        indices = np.frombuffer(f.read(nnz * 4), dtype=np.int32)

    return sp.csr_matrix((data, indices, indptr), shape=(num_rows, num_cols))

N_3d = 64

A_3d = construct_3d_poisson_matrix(N_3d)

# Display the shapes of the matrices
print(f"Shape of the 3D Poisson matrix: {A_3d.shape}, dtype: {A_3d.dtype}")

# Save matrices as binary files
directory = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/matrixA_poisson"
os.makedirs(directory, exist_ok=True)
save_matrix_as_bin(A_3d, os.path.join(directory, "A_3d64.bin"),'f')

A_3d_loaded = readA_sparse(N_3d, os.path.join(directory, "A_3d64.bin"), 'f')
print(f"Loaded matrix A with shape: {A_3d_loaded.shape}, dtype: {A_3d_loaded.dtype}")

print("\nVerifying loaded 3D Poisson matrix:")
verify_poisson_matrix(A_3d_loaded)

# Assert that the matrices are equal
assert np.allclose(A_3d_loaded.data, A_3d.data), "Data arrays are not equal!"
assert np.array_equal(A_3d_loaded.indices, A_3d.indices), "Indices arrays are not equal!"
assert np.array_equal(A_3d_loaded.indptr, A_3d.indptr), "Indptr arrays are not equal!"
assert A_3d_loaded.shape == A_3d.shape, "Matrix shapes are not equal!"
print("The loaded matrix matches the original matrix.")