import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.sparse as sparse
from numpy.linalg import norm
import time
import logging
import argparse

# Setting up directory paths for importing custom modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../../lib'))
sys.path.insert(1, lib_path)

import minres as mr
import helper_functions as hf

# Import Numba for optimized matrix multiplication
from numba import njit, prange

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
parser.add_argument("--plot", action='store_true',
                    help="Enable plotting of frequency spectra for verification")
args = parser.parse_args()

# %% Convert parsed arguments to variables
N = args.resolution
num_ritz_vectors = args.number_of_base_ritz_vectors
small_matmul_size = args.small_matmul_size

# Ensure the output directory exists
import pathlib

pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

dim_b = N ** 3 // 2

# %% Load the matrix A
if N == 64:
    A_file_name = os.path.join(args.dataset_dir, "tridiagA_3d.bin")
elif N == 128:
    A_file_name = os.path.join(args.dataset_dir, "tridiagA_3d.bin")
print("Start loading matrix A from " + A_file_name)
A = hf.readA_sparse_from_bin(dim_b, A_file_name, 'f')
print(f"Shape of matrix A: {A.shape}, dtype: {A.dtype}")

# Initialize MINRES solver with the matrix A
MR = mr.MINRESSparse(A)

# %% Generate a random vector for Lanczos iteration
rand_vec_x = np.random.normal(0, 1, [dim_b])
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
plt.figure()
plt.hist(ritz_vals, bins=50, color='skyblue', edgecolor='k')
plt.title("Histogram of Ritz Eigenvalues")
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.show()
print("Min Ritz eigenvalue:", np.min(ritz_vals))
print("Max Ritz eigenvalue:", np.max(ritz_vals))
print("Mean Ritz eigenvalue:", np.mean(ritz_vals))
print("Median Ritz eigenvalue:", np.median(ritz_vals))
print("Std Ritz eigenvalue:", np.std(ritz_vals))

# %% Transform back to the full space using Ritz vectors
ritz_vectors = np.matmul(W.transpose(), Q0).transpose()

if ritz_vectors.shape[1] > dim_b:
    ritz_vectors = ritz_vectors[:, :dim_b]

print(f"Ritz vectors shape: {ritz_vectors.shape}")
# Identify negative eigenvalues if you want to do special weighting
neg_indices = np.where(ritz_vals < 0)[0]
pos_indices = np.where(ritz_vals >= 0)[0]
print(f"Found {len(neg_indices)} negative, {len(pos_indices)} non-negative out of {len(ritz_vals)}.")

# %% For fast matrix multiply using Numba
@njit(parallel=True)
def mat_mult(A, B):
    assert A.shape[1] == B.shape[0]
    res = np.zeros((A.shape[0], B.shape[1]), )
    for i in prange(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(B.shape[1]):
                res[i, j] += A[i, k] * B[k, j]
    return res

# %% Fourier Filtering Functions
def apply_low_pass_filter(vector, frequency_cutoff):
    """
    Applies a low-pass filter to a 1D vector in the frequency domain.
    """
    freq_vector = np.fft.fft(vector)
    frequencies = np.fft.fftfreq(len(vector))
    mask = np.abs(frequencies) <= frequency_cutoff
    filtered_freq_vector = freq_vector * mask
    filtered_vector = np.fft.ifft(filtered_freq_vector)
    return np.real(filtered_vector)

def apply_high_pass_filter(vector, frequency_cutoff):
    """
    Applies a high-pass filter to a 1D vector in the frequency domain.
    """
    freq_vector = np.fft.fft(vector)
    frequencies = np.fft.fftfreq(len(vector))
    mask = np.abs(frequencies) > frequency_cutoff
    filtered_freq_vector = freq_vector * mask
    filtered_vector = np.fft.ifft(filtered_freq_vector)
    return np.real(filtered_vector)

# %% Prepare for Dataset Generation with 80% Training and 20% Validation
total_samples = args.sample_size  # e.g., 20000
small_matmul_size = args.small_matmul_size  # e.g., 200
if total_samples % small_matmul_size != 0:
    raise ValueError("Total samples must be divisible by small_matmul_size for batch processing.")
total_batches = total_samples // small_matmul_size  # e.g., 100

training_percentage = 0.8  # 80% Training
training_samples = int(total_samples * training_percentage)
validation_samples = total_samples - training_samples

train_low_freq_samples = training_samples // 2
train_high_freq_samples = training_samples - train_low_freq_samples
val_low_freq_samples = validation_samples // 2
val_high_freq_samples = validation_samples - val_low_freq_samples


# Create separate directories for Training and Validation datasets
train_dir = os.path.join(args.output_dir, 'train')
val_dir = os.path.join(args.output_dir, 'val')
pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(val_dir).mkdir(parents=True, exist_ok=True)

# %% Identify and Exclude Zero Ritz Values
num_zero_ritz_vals = 0
while num_zero_ritz_vals < num_ritz_vectors and abs(ritz_vals[num_zero_ritz_vals]) < 1.0e-12:
    num_zero_ritz_vals += 1
print(f"Number of zero (or near-zero) Ritz values excluded: {num_zero_ritz_vals}")

# %% Generate the Dataset with Fourier Filtering
print("Creating Dataset with Fourier Transformation")

# Parameters for filters
low_pass_cutoff_ratio = 0.1  # Retain lower 10% frequencies for low-pass filter
high_pass_cutoff_ratio = 0.1  # Retain higher 90% frequencies for high-pass filter

# Total number of batches for training and validation
train_batches = training_samples // small_matmul_size  # 16000 / 200 = 80 batches
val_batches = validation_samples // small_matmul_size  # 4000 / 200 = 20 batches

# Calculate the number of batches for low and high-frequency data
train_low_freq_batches = train_low_freq_samples // small_matmul_size  # 8000 / 200 = 40 batches
train_high_freq_batches = train_high_freq_samples // small_matmul_size  # 8000 / 200 = 40 batches

val_low_freq_batches = val_low_freq_samples // small_matmul_size  # 2000 / 200 = 10 batches
val_high_freq_batches = val_high_freq_samples // small_matmul_size  # 2000 / 200 = 10 batches

print("val_low_freq_samples:", val_low_freq_samples)
print("val_high_freq_samples:", val_high_freq_samples)
print("val_low_freq_batches:", val_low_freq_batches)
print("val_high_freq_batches:", val_high_freq_batches)
print("train_low_freq_samples:", train_low_freq_samples)
print("train_high_freq_samples:", train_high_freq_samples)
print("train_low_freq_batches:", train_low_freq_batches)
print("train_high_freq_batches:", train_high_freq_batches)

def generate_coef_matrix(num_ritz_used, batch_size, neg_indices, pos_indices,
                           neg_factor=9.0, weight_neg_only=False):
    """
    Generates a [num_ritz_used x batch_size] coefficient matrix.
    
    If weight_neg_only is True, then only the rows corresponding to negative eigenvalues
    are nonzero (or are amplified), while the other rows are set to zero in a fraction
    of the columns. Otherwise, all rows are sampled from a normal distribution,
    and rows corresponding to negative eigenvalues are multiplied by neg_factor.
    """
    # Start with a normal random sample for all Ritz vectors.
    coef_matrix = np.random.normal(0, 1, (num_ritz_used, batch_size))
    
    if weight_neg_only:
        # For a fraction of columns (say, fraction_neg), force the coefficients for positive
        # eigenvalue directions to zero.
        fraction_neg = 0.1  # For example, 10% of the columns are purely negative.
        num_neg_cols = int(fraction_neg * batch_size)
        if num_neg_cols > 0 and len(neg_indices) > 0:
            # Zero out the rows for positive eigenvalue indices for those columns.
            coef_matrix[pos_indices, :num_neg_cols] = 0.0
            # Optionally, further amplify negative indices in these columns.
            coef_matrix[neg_indices, :num_neg_cols] *= neg_factor
    else:
        # Multiply negative eigenvalue rows by the neg_factor uniformly.
        if len(neg_indices) > 0:
            coef_matrix[neg_indices, :] *= neg_factor
    
    return coef_matrix

def generate_and_save_batches(num_batches, batch_size, freq_type, data_dir, start_index,
                              ritz_vectors, num_ritz_vectors, num_zero_ritz_vals,
                              neg_indices, pos_indices, neg_factor=9.0, weight_neg_only=False):
    """
    Generates and saves data batches using Fourier filtering. This version applies
    extra weighting to negative eigenvalue directions in the coefficient matrix.
    """
    sample_index = start_index
    for batch_num in range(num_batches):
        t0 = time.time()
        # Generate random coefficients from the Ritz space (excluding near-zero ones)
        num_ritz_used = num_ritz_vectors - num_zero_ritz_vals
        coef_matrix = generate_coef_matrix(num_ritz_used, batch_size, neg_indices, pos_indices,
                                             neg_factor=neg_factor, weight_neg_only=weight_neg_only)
        
        # Map the coefficients to x by forming a linear combination of Ritz vectors.
        # ritz_vectors has shape (num_ritz_vectors, dim_b); we use rows from num_zero_ritz_vals onward.
        x_temp = mat_mult(ritz_vectors[num_zero_ritz_vals:num_ritz_vectors, :].transpose(),
                          coef_matrix).transpose()  # shape (batch_size, dim_b)
        
        # Normalize each sample in x_temp
        epsilon = 1e-10
        norms = np.linalg.norm(x_temp, axis=1, keepdims=True) + epsilon
        x_temp_normalized = x_temp / norms
        
        # Apply Fourier filtering: this step is independent of the coefficient weighting.
        for i in range(batch_size):
            if freq_type == 'low':
                x_temp_normalized[i] = apply_low_pass_filter(x_temp_normalized[i], low_pass_cutoff_ratio)
            elif freq_type == 'high':
                x_temp_normalized[i] = apply_high_pass_filter(x_temp_normalized[i], high_pass_cutoff_ratio)
            else:
                raise ValueError("freq_type must be 'low' or 'high'")
        
        # Compute b = A * x for each sample
        b_rhs_temp = np.zeros((batch_size, dim_b), dtype=np.float32)
        for i in range(batch_size):
            b_rhs_temp[i] = A.dot(x_temp_normalized[i])
        
        # Construct the augmented right-hand side vector d = [b; 0] of length N^3
        d_rhs_temp = np.zeros((batch_size, N ** 3), dtype=np.float32)
        for i in range(batch_size):
            d_rhs_temp[i, :dim_b] = b_rhs_temp[i]
        
        # Save each sample to file
        for i in range(batch_size):
            d_filename = os.path.join(data_dir, f'd_{sample_index}.npy')
            np.save(d_filename, d_rhs_temp[i].astype(np.float32))
            sample_index += 1
        
        t1 = time.time()
        print(f"Batch {batch_num + 1}/{num_batches} for {freq_type}-frequency data saved in {t1 - t0:.2f} seconds.")
    return sample_index



# %% Generate Training Data
print("Generating Training Data...")
sample_index = 0  # Start index for training samples

# Generating 8,000 Low-Frequency Training Samples:
print("Generating low-frequency training data...")
sample_index = generate_and_save_batches(train_low_freq_batches, small_matmul_size, 'low', train_dir, sample_index, ritz_vectors, 
    num_ritz_vectors, 
    num_zero_ritz_vals, 
    neg_indices, 
    pos_indices, 
    neg_factor=12.0, 
    weight_neg_only=False
)

# Generating 8,000 High-Frequency Training Samples:
print("Generating high-frequency training data...")
sample_index = generate_and_save_batches(train_high_freq_batches, small_matmul_size, 'high', train_dir, sample_index, ritz_vectors, 
    num_ritz_vectors, 
    num_zero_ritz_vals, 
    neg_indices, 
    pos_indices, 
    neg_factor=12.0, 
    weight_neg_only=False
)
# After training data generation, sample_index should be 16000

# %% Generate Validation Data
print("Generating Validation Data...")
sample_index = 16000  # Start index for validation samples (adjusted as per your requirement)

# Generating 2,000 Low-Frequency Validation Samples:
print("Generating low-frequency validation data...")
sample_index = generate_and_save_batches(val_low_freq_batches, small_matmul_size, 'low', val_dir, sample_index,
    ritz_vectors, 
    num_ritz_vectors, 
    num_zero_ritz_vals, 
    neg_indices, 
    pos_indices, 
    neg_factor=12.0, 
    weight_neg_only=False
)

# Generating 2,000 High-Frequency Validation Samples:
print("Generating high-frequency validation data...")
sample_index = generate_and_save_batches(val_high_freq_batches, small_matmul_size, 'high', val_dir, sample_index,
    ritz_vectors, 
    num_ritz_vectors, 
    num_zero_ritz_vals, 
    neg_indices, 
    pos_indices, 
    neg_factor=12.0, 
    weight_neg_only=False
)

print("Dataset generation completed.")

# lstsquares_problem normalized