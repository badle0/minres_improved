# This is an example test code for the paper
# %% Load the required libraries
import sys
import os
import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
import time
import argparse

# Import custom modules
# Setting up directory paths for importing custom modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../lib'))
sys.path.insert(1, lib_path)
import minres as mr
import helper_functions as hf

# %% Get Arguments from parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution", type=int, choices=[64, 128, 256],
                    help="N or resolution of test", default=64)
parser.add_argument("-k", "--trained_model_type", type=int, choices=[64, 128],
                    help="which model to test", default=64)
parser.add_argument("-f", "--float_type", type=int, choices=[16, 32],
                    help="model parameters' float type", default=32)
parser.add_argument("--model_index", type=int,
                    help="model parameters")
parser.add_argument("-ex", "--example_type", type=str,
                    help="example type", default="smoke_passing_bunny")
parser.add_argument("-fn", "--frame_number", type=int,
                    help="which frame in sim to test", default=1)
parser.add_argument("--max_iter", type=int,
                    help="maximum iteration of algorithms", default=1000)
parser.add_argument("-tol", "--tolerance", type=float,
                    help="tolerance for both DEEP MINRES and MINRES algorithm", default=1.0e-4)
parser.add_argument("--verbose_deepmr", type=bool,
                    help="prints residuals of MINRES algorithm for each iteration", default=False)
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="path to the dataset")
parser.add_argument("--trained_model_dir", type=str, required=True,
                    help="path to the trained models")
parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str,
                    help="Determines if DEEP MINRES uses GPU. Default DEEP MINRES only uses CPU.", default="")
parser.add_argument('--skip_deepmr', action="store_true",
                    help='skips DEEP MINRES tests')
parser.add_argument('--skip_mr', action="store_true",
                    help='skips minres test')
parser.add_argument('--skip_dpmr', action="store_true",
                    help='skips preconditioned minres test')
parser.add_argument('--skip_ilupmr', action="store_true",
                    help='skips incomplete lu preconditioned minres test')
parser.add_argument('--skip_icpmr', action="store_true",
                    help='skips incomplete cholesky preconditioned minres test')

args = parser.parse_args()

# %%
N = args.resolution
k = args.trained_model_type
if N == 64:
    print("For N=64 resolution there is only 64-trained model.")
    k = 64
float_type = args.float_type
if float_type == 16:
    dtype_ = tf.float16
if float_type == 32:
    dtype_ = tf.float32
model_index = args.model_index
example_name = args.example_type
frame_number = args.frame_number
max_iter = args.max_iter
tol = args.tolerance
verbose_deepmr = args.verbose_deepmr
dataset_path = args.dataset_dir
trained_model_path = args.trained_model_dir
# This determines if DEEPMR uses only CPU or uses GPU. By default it only uses CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

# %%
if example_name in ["smoke_plume", "smoke_passing_bunny"]:
    matrix_frame_number = 1
else:
    matrix_frame_number = frame_number

trained_model_name = trained_model_path + "/saved_models/3D_N" + str(k) + "_json_E" + str(model_index) + "/"

# %% Getting RHS for the Testing
d_type = np.float32

def get_vector_from_source(file_rhs, d_type=np.float32):
    if (os.path.exists(file_rhs)):
        return_vector = np.fromfile(file_rhs, dtype=d_type)
        return return_vector
    else:
        print("RHS does not exist at " + file_rhs)


print("Matrix A and rhs b is loading...")
initial_normalization = False
b_file_name = dataset_path + "/lstsquareproblem/test_dataset/" + "d_test3.bin"
A_file_name = dataset_path + "/lstsquareproblem/test_dataset/" + "S_test3.bin"
A = hf.readA_sparse(N, A_file_name, 'f')
b = get_vector_from_source(b_file_name)
MR = mr.MINRESSparse(A)
normalize_ = False
max_mr_iter = 1000

# %% Testing
if not args.skip_deepmr:
    print("Loading the model...")
    # for Poisson Matrix example, the best performing model is epoch 43
    trained_model_name = trained_model_path + "/saved_models/3D_N" + str(k) + "_json_E" + str(
        model_index) + "/"
    model = hf.load_model_from_source(trained_model_name)
    model.summary()
    print("Model loaded. Number of parameters in the model is ", model.count_params())
    model_predict = lambda r: model(tf.convert_to_tensor(r.reshape([1, N, N, N]), dtype=dtype_),
                                    training=False).numpy()[0, :, :].reshape([N ** 3])
    # Dummy Calling
    model_predict(b)

    print("DEEPMINRES is running...")
    t0 = time.time()
    max_mr_iter = 1000
    x_sol, res_arr = MR.deepminres(b, np.zeros(b.shape), model_predict, max_mr_iter, tol, verbose_deepmr)
    time_cg_ml = time.time() - t0
    print("DEEPMINRES took ", time_cg_ml, " secs.")
    
if not args.skip_mr:
    print("MINRES is running...")
    t0 = time.time()
    x_sol_mr, res_arr_mr = MR.minres(b, np.zeros(b.shape), max_mr_iter, tol, True)
    time_cg = time.time() - t0
    print("MINRES took ", time_cg, " secs")

if not args.skip_dpmr:
    print("DiagonalPMR is running...")
    t0 = time.time()
    M_inv = MR.create_diagonal_preconditioner()

    def ic_precond(x):
        return M_inv.multiply_A(x)

    x_sol_mr, res_arr_mr = MR.pmr_normal(b, np.zeros(b.shape), ic_precond, max_mr_iter, tol, verbose_deepmr)
    time_cg = time.time() - t0
    print("DiagonalPMR took ", time_cg, " secs")

if not args.skip_ilupmr:
    LiLUmr_test_folder = dataset_path + "/test_dataset/N64/" + "L.npz"
    UiLUmr_test_folder = dataset_path + "/test_dataset/N64/" + "U.npz"
    L = sparse.load_npz(LiLUmr_test_folder)
    U = sparse.load_npz(UiLUmr_test_folder)

    def ic_precond(x):
        y_inter = sparse.linalg.spsolve_triangular(L,x, lower=True) #Forward sub
        return sparse.linalg.spsolve_triangular(U,y_inter, lower=False) #backward sub

    print("IncompleteLUPMR is running...")
    t0 = time.time()
    x_sol_mr, res_arr_mr = MR.pmr_normal(b, np.zeros(b.shape), ic_precond, max_mr_iter, tol, verbose_deepmr)
    time_iLUmr = time.time() - t0
    print("IncompleteLUPMR took ", time_iLUmr, " secs")

if not args.skip_icpmr:
    def create_boundary_mask(n):
        """Create a mask to exclude the boundary nodes of an n^3 grid."""
        mask = np.ones((n, n, n), dtype=bool)
        mask[0, :, :] = mask[-1, :, :] = False
        mask[:, 0, :] = mask[:, -1, :] = False
        mask[:, :, 0] = mask[:, :, -1] = False
        return mask.flatten()

    # Create a mask to exclude boundary nodes
    boundary_mask = create_boundary_mask(N)

    # Apply the mask to filter the Poisson matrix and RHS vector
    A2 = A[boundary_mask, :][:, boundary_mask]
    b2 = b[boundary_mask]
    MR2 = mr.MINRESSparse(A2)

    icpmr_test_folder = dataset_path + "/test_dataset/N" + str(N) + "/L1.npz"
    L = sparse.load_npz(icpmr_test_folder)

    def ic_precond(x):
        y_inter = sparse.linalg.spsolve_triangular(L,x, lower=True) #Forward sub
        return sparse.linalg.spsolve_triangular(L.transpose(),y_inter, lower=False) #backward sub

    print("IncompleteCholeskyPMR is running...")
    t0 = time.time()
    x_sol_mr, res_arr_mr = MR2.pmr_normal(b2, np.zeros(b2.shape), ic_precond, max_mr_iter, tol, verbose_deepmr)
    time_icpmr = time.time() - t0
    print("IncompleteChleskyPMR took ", time_icpmr, " secs")