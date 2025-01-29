#!/usr/bin/env python3
import os
import struct
import numpy as np
import scipy.sparse as sp
from scipy import sparse

###################################
# Helper Functions
###################################

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

def idx_3d(i, j, k, N):
    return i*(N**2) + j*N + k

def create_vector_b(A):
    n = A.shape[0]
    indices = np.arange(n)
    smooth_part = np.sin(2 * np.pi * indices / n).astype(np.float32)
    noise_part = 0.1 * np.random.normal(0,1,n).astype(np.float32)
    b = smooth_part + noise_part
    if np.linalg.norm(b) < 1e-10:
        b += 0.01*np.random.normal(0,1,n).astype(np.float32)
    return b

###################################
# Scenarios
###################################

# 1. Variable Coefficients
def create_variable_coeff_3d_poisson(N):
    np.random.seed(42)
    dx = 1.0/(N-1)
    a_field = 0.5 + np.random.rand(N,N,N).astype(np.float32)

    row, col, data = [], [], []

    def a_xp(i,j,k):
        return 0.5*(a_field[i,j,k]+a_field[i+1,j,k]) if i < N-1 else a_field[i,j,k]
    def a_xm(i,j,k):
        return 0.5*(a_field[i,j,k]+a_field[i-1,j,k]) if i > 0 else a_field[i,j,k]
    def a_yp(i,j,k):
        return 0.5*(a_field[i,j,k]+a_field[i,j+1,k]) if j < N-1 else a_field[i,j,k]
    def a_ym(i,j,k):
        return 0.5*(a_field[i,j,k]+a_field[i,j-1,k]) if j > 0 else a_field[i,j,k]
    def a_zp(i,j,k):
        return 0.5*(a_field[i,j,k]+a_field[i,j,k+1]) if k < N-1 else a_field[i,j,k]
    def a_zm(i,j,k):
        return 0.5*(a_field[i,j,k]+a_field[i,j,k-1]) if k > 0 else a_field[i,j,k]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                current = idx_3d(i,j,k,N)
                diag_val = 0.0
                if k > 0:
                    val = -(a_zm(i,j,k)/dx**2)
                    row.append(current); col.append(idx_3d(i,j,k-1,N)); data.append(val)
                    diag_val -= val
                if k < N-1:
                    val = -(a_zp(i,j,k)/dx**2)
                    row.append(current); col.append(idx_3d(i,j,k+1,N)); data.append(val)
                    diag_val -= val

                if j > 0:
                    val = -(a_ym(i,j,k)/dx**2)
                    row.append(current); col.append(idx_3d(i,j,k-1,N)); data.append(val)
                    diag_val -= val
                    # Correction: line above incorrectly uses (i,j,k-1) instead of (i,j-1,k)
                    # Let's fix that:
                    row[-1], col[-1] = current, idx_3d(i,j-1,k,N)
                if j < N-1:
                    val = -(a_yp(i,j,k)/dx**2)
                    row.append(current); col.append(idx_3d(i,j+1,k,N)); data.append(val)
                    diag_val -= val

                if i > 0:
                    val = -(a_xm(i,j,k)/dx**2)
                    row.append(current); col.append(idx_3d(i-1,j,k,N)); data.append(val)
                    diag_val -= val
                if i < N-1:
                    val = -(a_xp(i,j,k)/dx**2)
                    row.append(current); col.append(idx_3d(i+1,j,k,N)); data.append(val)
                    diag_val -= val

                row.append(current); col.append(current); data.append(diag_val)

    A = sp.coo_matrix((data,(row,col)), shape=(N**3,N**3), dtype=np.float32)
    A = 0.5*(A + A.transpose())
    A = A + sp.eye(N**3, dtype=np.float32)*1e-3
    return A.tocsr()

# 2. Strong Anisotropy
def create_anisotropic_3d_poisson(N, anisotropy=(1.0,0.01,0.001)):
    dx = 1.0/(N-1)
    ax, ay, az = anisotropy

    row, col, data = [], [], []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                current = idx_3d(i,j,k,N)
                diag_val = 0.0
                if k > 0:
                    val = -az/dx**2
                    row.append(current); col.append(idx_3d(i,j,k-1,N)); data.append(val)
                    diag_val -= val
                if k < N-1:
                    val = -az/dx**2
                    row.append(current); col.append(idx_3d(i,j,k+1,N)); data.append(val)
                    diag_val -= val
                if j > 0:
                    val = -ay/dx**2
                    row.append(current); col.append(idx_3d(i,j-1,k,N)); data.append(val)
                    diag_val -= val
                if j < N-1:
                    val = -ay/dx**2
                    row.append(current); col.append(idx_3d(i,j+1,k,N)); data.append(val)
                    diag_val -= val
                if i > 0:
                    val = -ax/dx**2
                    row.append(current); col.append(idx_3d(i-1,j,k,N)); data.append(val)
                    diag_val -= val
                if i < N-1:
                    val = -ax/dx**2
                    row.append(current); col.append(idx_3d(i+1,j,k,N)); data.append(val)
                    diag_val -= val

                row.append(current); col.append(current); data.append(diag_val)

    A = sp.coo_matrix((data,(row,col)), shape=(N**3,N**3), dtype=np.float32)
    A = 0.5*(A + A.transpose())
    A = A + sp.eye(N**3, dtype=np.float32)*1e-3
    return A.tocsr()

# 3. Non-Uniform Grid
def create_nonuniform_3d_poisson(N):
    z = np.linspace(0,1,N)**2
    dx = 1.0/(N-1)
    row, col, data = [], [], []

    def dz(k):
        if k == 0 or k == N-1:
            return dx
        return z[k]-z[k-1]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                current = idx_3d(i,j,k,N)
                diag_val = 0.0
                dz_val = dz(k)
                dx2 = dx**2

                if k > 0:
                    val = -1/dz_val**2
                    row.append(current); col.append(idx_3d(i,j,k-1,N)); data.append(val)
                    diag_val -= val
                if k < N-1:
                    val = -1/dz_val**2
                    row.append(current); col.append(idx_3d(i,j,k+1,N)); data.append(val)
                    diag_val -= val

                if j > 0:
                    val = -1/dx2
                    row.append(current); col.append(idx_3d(i,j-1,k,N)); data.append(val)
                    diag_val -= val
                if j < N-1:
                    val = -1/dx2
                    row.append(current); col.append(idx_3d(i,j+1,k,N)); data.append(val)
                    diag_val -= val

                if i > 0:
                    val = -1/dx2
                    row.append(current); col.append(idx_3d(i-1,j,k,N)); data.append(val)
                    diag_val -= val
                if i < N-1:
                    val = -1/dx2
                    row.append(current); col.append(idx_3d(i+1,j,k,N)); data.append(val)
                    diag_val -= val

                row.append(current); col.append(current); data.append(diag_val)

    A = sp.coo_matrix((data,(row,col)), shape=(N**3,N**3), dtype=np.float32)
    A = 0.5*(A + A.transpose())
    A = A + sp.eye(N**3, dtype=np.float32)*1e-3
    return A.tocsr()

# 4. Mixed Boundary Conditions
def create_mixed_bc_3d_poisson(N):
    dx = 1.0/(N-1)
    row, col, data = [], [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                current = idx_3d(i,j,k,N)
                diag_val = 0.0

                if k > 0:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j,k-1,N)); data.append(val)
                    diag_val -= val
                if k < N-1:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j,k+1,N)); data.append(val)
                    diag_val -= val

                if j > 0:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j-1,k,N)); data.append(val)
                    diag_val -= val
                if j < N-1:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j+1,k,N)); data.append(val)
                    diag_val -= val

                # Neumann at i=0 and i=N-1: no x neighbors at boundaries
                if 0 < i < N-1:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i-1,j,k,N)); data.append(val)
                    diag_val -= val
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i+1,j,k,N)); data.append(val)
                    diag_val -= val

                row.append(current); col.append(current); data.append(diag_val)

    A = sp.coo_matrix((data,(row,col)), shape=(N**3,N**3), dtype=np.float32)
    A = 0.5*(A + A.transpose())
    A = A + sp.eye(N**3, dtype=np.float32)*1e-3
    return A.tocsr()

# 5. 27-Point Stencil
def create_27point_stencil_3d_poisson(N):
    dx = 1.0/(N-1)
    offsets = [-1,0,1]
    val_neighbor = -1/(26*(dx**2))  # distribute evenly

    row, col, data = [], [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                current = idx_3d(i,j,k,N)
                neighbors = []
                for di in offsets:
                    for dj in offsets:
                        for dk in offsets:
                            if di==0 and dj==0 and dk==0:
                                continue
                            ni, nj, nk = i+di,j+dj,k+dk
                            if 0<=ni<N and 0<=nj<N and 0<=nk<N:
                                neighbors.append((ni,nj,nk))

                diag_val = 0.0
                for (ni,nj,nk) in neighbors:
                    nidx = idx_3d(ni,nj,nk,N)
                    row.append(current); col.append(nidx); data.append(val_neighbor)
                    diag_val -= val_neighbor

                row.append(current); col.append(current); data.append(diag_val)

    A = sp.coo_matrix((data,(row,col)), shape=(N**3,N**3), dtype=np.float32)
    A = 0.5*(A + A.transpose())
    A = A + sp.eye(N**3, dtype=np.float32)*1e-3
    return A.tocsr()

###################################
# Additional Larger Problem Scenarios
###################################

# Standard Poisson (6-point stencil) for bigger grids
def create_standard_poisson_3d(N):
    dx = 1.0/(N-1)
    row, col, data = [], [], []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                current = idx_3d(i,j,k,N)
                diag_val = 0.0

                if k > 0:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j,k-1,N)); data.append(val)
                    diag_val -= val
                if k < N-1:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j,k+1,N)); data.append(val)
                    diag_val -= val

                if j > 0:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j-1,k,N)); data.append(val)
                    diag_val -= val
                if j < N-1:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i,j+1,k,N)); data.append(val)
                    diag_val -= val

                if i > 0:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i-1,j,k,N)); data.append(val)
                    diag_val -= val
                if i < N-1:
                    val = -1/dx**2
                    row.append(current); col.append(idx_3d(i+1,j,k,N)); data.append(val)
                    diag_val -= val

                row.append(current); col.append(current); data.append(diag_val)

    A = sp.coo_matrix((data,(row,col)), shape=(N**3,N**3), dtype=np.float32)
    A = 0.5*(A + A.transpose())
    A = A + sp.eye(N**3, dtype=np.float32)*1e-3
    return A.tocsr()


if __name__ == "__main__":
    base_dir = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/poisson_test_dataset"
    os.makedirs(base_dir, exist_ok=True)

    N = 64  # for the first five scenarios

    # 1. Variable Coefficients
    var_coeff_dir = os.path.join(base_dir, "variable_coeff")
    os.makedirs(var_coeff_dir, exist_ok=True)
    A_var = create_variable_coeff_3d_poisson(N)
    b_var = create_vector_b(A_var)
    save_matrix_as_bin(A_var, os.path.join(var_coeff_dir, "A_variable_coeff.bin"), 'f')
    save_vector_as_bin(b_var, os.path.join(var_coeff_dir, "b_variable_coeff.bin"), 'f')

    # 2. Strong Anisotropy
    anisotropy_dir = os.path.join(base_dir, "strong_anisotropy")
    os.makedirs(anisotropy_dir, exist_ok=True)
    A_ani = create_anisotropic_3d_poisson(N, (1.0,0.01,0.001))
    b_ani = create_vector_b(A_ani)
    save_matrix_as_bin(A_ani, os.path.join(anisotropy_dir, "A_anisotropic.bin"), 'f')
    save_vector_as_bin(b_ani, os.path.join(anisotropy_dir, "b_anisotropic.bin"), 'f')

    # 3. Non-Uniform Grid
    nonuniform_dir = os.path.join(base_dir, "nonuniform_grid")
    os.makedirs(nonuniform_dir, exist_ok=True)
    A_non = create_nonuniform_3d_poisson(N)
    b_non = create_vector_b(A_non)
    save_matrix_as_bin(A_non, os.path.join(nonuniform_dir, "A_nonuniform.bin"), 'f')
    save_vector_as_bin(b_non, os.path.join(nonuniform_dir, "b_nonuniform.bin"), 'f')

    # 4. Mixed Boundary Conditions
    mixed_bc_dir = os.path.join(base_dir, "mixed_bc")
    os.makedirs(mixed_bc_dir, exist_ok=True)
    A_mbc = create_mixed_bc_3d_poisson(N)
    b_mbc = create_vector_b(A_mbc)
    save_matrix_as_bin(A_mbc, os.path.join(mixed_bc_dir, "A_mixed_bc.bin"), 'f')
    save_vector_as_bin(b_mbc, os.path.join(mixed_bc_dir, "b_mixed_bc.bin"), 'f')

    # 5. 27-Point Stencil
    stencil_dir = os.path.join(base_dir, "27point_stencil")
    os.makedirs(stencil_dir, exist_ok=True)
    A_27 = create_27point_stencil_3d_poisson(N)
    b_27 = create_vector_b(A_27)
    save_matrix_as_bin(A_27, os.path.join(stencil_dir, "A_27point.bin"), 'f')
    save_vector_as_bin(b_27, os.path.join(stencil_dir, "b_27point.bin"), 'f')

    # Additional bigger grids scenarios:
    # N=128
    bigger_128_dir = os.path.join(base_dir, "bigger_128")
    os.makedirs(bigger_128_dir, exist_ok=True)
    A_128 = create_standard_poisson_3d(128)
    b_128 = create_vector_b(A_128)
    save_matrix_as_bin(A_128, os.path.join(bigger_128_dir, "A_128.bin"), 'f')
    save_vector_as_bin(b_128, os.path.join(bigger_128_dir, "b_128.bin"), 'f')

    # N=256
    # bigger_256_dir = os.path.join(base_dir, "bigger_256")
    # os.makedirs(bigger_256_dir, exist_ok=True)
    # A_256 = create_standard_poisson_3d(256)
    # b_256 = create_vector_b(A_256)
    # save_matrix_as_bin(A_256, os.path.join(bigger_256_dir, "A_256.bin"), 'f')
    # save_vector_as_bin(b_256, os.path.join(bigger_256_dir, "b_256.bin"), 'f')

    print("All requested test datasets have been generated.")
