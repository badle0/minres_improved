import os
import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
print("Built with CUDA support: ", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import gc
import scipy.sparse as sparse
import time
import argparse

# Setting up directory paths for importing custom modules
dir_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(dir_path, '../../lib'))
sys.path.insert(1, lib_path)

import minres as mr
import helper_functions as hf

# %% Get Arguments from parser
parser = argparse.ArgumentParser()

parser.add_argument("-N", "--resolution", type=int, choices=[64, 128],
                    help="N or resolution of test", default=64)

parser.add_argument("--total_number_of_epochs", type=int,
                    help="Total number of epochs for training", default=1000)

parser.add_argument("--epoch_each_number", type=int,
                    help="epoch number of", default=1)

parser.add_argument("--batch_size", type=int,
                    help="--batch_size.", default=10)

parser.add_argument("--loading_number", type=int,
                    help="loading number of each iteration", default=100)

parser.add_argument("--gpu_usage", type=int,
                    help="gpu usage, in terms of GB.", default=3)

parser.add_argument("--gpu_idx", type=str,
                    help="which gpu to use.", default='1')

parser.add_argument("--data_dir", type=str,
                    help="path to the folder containing dataset vectors", default='../2024minresdata/')

args = parser.parse_args()

# Convert parsed arguments to variables
N = args.resolution
epoch_num = args.total_number_of_epochs
epoch_each_iter = args.epoch_each_number
b_size = args.batch_size
loading_number = args.loading_number
gpu_usage = args.gpu_usage * 1024 # Convert GB to MB
which_gpu = args.gpu_idx

project_name = "3D_N" + str(N)
project_folder_subname = os.path.basename(os.getcwd())
print("project_folder_subname = ", project_folder_subname)

project_folder_general = "lstsquares_filtered_training_fixed/3D_N" + str(N) + "/"
dim2 = N ** 3
lr = 1.0e-4 # Learning rate

# GPU configuration to limit memory usage
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(gpus[0],
                                                   [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_usage)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# Load sparse matrix A
if N == 64:
    A_file_name = args.data_dir + "/lstsquares_problem/matrixS_3d" + str(N) + ".bin"
elif N == 128:
    A_file_name = args.data_dir + "/matrixA/A_2d" + str(N) + ".bin"
A_sparse_scipy = hf.readA_sparse(N, A_file_name, 'f')
# Initialize the MINRES solver with the loaded sparse matrix A
MR = mr.MINRESSparse(A_sparse_scipy)
# Convert the sparse matrix A from SciPy's CSR format to COO (Coordinate list) format
# COO format is easier to work with when constructing TensorFlow's SparseTensor
coo = A_sparse_scipy.tocoo()
# Create an array of indices from the row and column indices of the non-zero elements in A
# This is necessary for constructing a TensorFlow SparseTensor
indices = np.mat([coo.row, coo.col]).transpose()
# Convert the sparse matrix A from SciPy format to TensorFlow's SparseTensor format
# TensorFlow requires a SparseTensor for sparse matrix operations in the neural network model
A_sparse = tf.SparseTensor(indices, np.float32(coo.data), coo.shape)

# Custom loss function
def custom_loss_function_cnn_1d_fast(y_true, y_pred):
    """
    Custom loss function for training the CNN model.
    It calculates the error between the true and predicted values using a sparse matrix multiplication approach.
    """
    b_size_ = len(y_true)
    err = 0
    for i in range(b_size):
        A_tilde_inv = 1 / tf.tensordot(tf.reshape(y_pred[i], [1, dim2]),
                                       tf.sparse.sparse_dense_matmul(A_sparse, tf.reshape(y_pred[i], [dim2, 1])),
                                       axes=1)
        qTb = tf.tensordot(tf.reshape(y_pred[i], [1, dim2]), tf.reshape(y_true[i], [dim2, 1]), axes=1)
        x_initial_guesses = tf.reshape(y_pred[i], [dim2, 1]) * qTb * A_tilde_inv
        err = err + tf.reduce_sum(tf.math.square(
            tf.reshape(y_true[i], [dim2, 1]) - tf.sparse.sparse_dense_matmul(A_sparse, x_initial_guesses)))
    return err / b_size_

# %% Define CNN model
dim = N
fil_num = 16 # Number of filters
input_rhs = keras.Input(shape=(dim, dim, dim, 1))

# model design
if N == 64:
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

    apa = layers.AveragePooling3D((2, 2, 2), padding='same')(lb)
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa

    upa = layers.UpSampling3D((2, 2, 2))(apa) + lb
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa)
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa)
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

    last_layer = layers.Dense(1, activation='linear')(upa)

elif N == 128:
    first_layer = layers.Conv3D(fil_num, (3, 3, 3), activation='linear', padding='same')(input_rhs)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(first_layer)
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)
    la = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(lb) + la
    lb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(la)

    apa = layers.AveragePooling3D((2, 2, 2), padding='same')(lb)  # 7
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa
    apb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apa)
    apa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(apb) + apa

    upa = layers.UpSampling3D((2, 2, 2))(apa) + lb
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa)
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa)
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa
    upb = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upa)
    upa = layers.Conv3D(fil_num, (3, 3, 3), activation='relu', padding='same')(upb) + upa

    last_layer = layers.Dense(1, activation='linear')(upa)

model = keras.Model(input_rhs, last_layer)
model.compile(optimizer="Adam", loss=custom_loss_function_cnn_1d_fast)
model.optimizer.lr = lr
model.summary()

# File paths for saving training and validation loss
training_loss_name = project_folder_general + project_folder_subname + "/" + project_name + "_training_loss.npy"
validation_loss_name = project_folder_general + project_folder_subname + "/" + project_name + "_validation_loss.npy"
training_loss = []
validation_loss = []

# rhs vector Dataset directory
if N == 64:
    foldername = args.data_dir + "/lstsquares_problem/training_dataset_fourier/"
elif N == 128:
    foldername = args.data_dir + "/training_dataset/"

# Define separate directories for Training and Validation datasets
train_dir = os.path.join(foldername, 'train')
val_dir = os.path.join(foldername, 'val')

# Number of total data points
train_sample_size = 16000  # d_0.npy to d_15999.npy
val_sample_size = 4000     # d_16000.npy to d_19999.npy
total_data_points = 20000
for_loading_number = round(total_data_points / loading_number)
# b_rhs = np.zeros([loading_number, dim2])

# %% Training

for i in range(1, epoch_num + 1):
    print("Training at epoch = " + str(i))

    training_loss_inner = []
    validation_loss_inner = []
    t0 = time.time()

    # Shuffle training and validation indices separately
    perm_train = np.random.permutation(train_sample_size)  # 16,000
    perm_val = np.random.permutation(val_sample_size)  # 4,000

    for ii in range(for_loading_number):  # 200 sub-training sessions
        print(f"Sub-training session {ii + 1}/{for_loading_number} at epoch {i}")

        # Initialize arrays to hold training and validation samples
        b_train = np.zeros([80, dim2], dtype=np.float32)
        b_val = np.zeros([20, dim2], dtype=np.float32)

        # Load 80 training samples
        for j in range(80):
            train_idx = perm_train[80 * ii + j]
            train_file = os.path.join(train_dir, f'd_{train_idx}.npy')
            try:
                with open(train_file, 'rb') as f:
                    b_train[j] = np.load(f)
            except FileNotFoundError:
                print(f"Training file {train_file} not found.")
                continue
            except Exception as e:
                print(f"Error loading training file {train_file}: {e}")
                continue

        # Load 20 validation samples
        for j in range(20):
            val_idx = perm_val[20 * ii + j]
            actual_val_idx = 16000 + val_idx
            val_file = os.path.join(val_dir, f'd_{actual_val_idx}.npy')
            try:
                with open(val_file, 'rb') as f:
                    b_val[j] = np.load(f)
            except FileNotFoundError:
                print(f"Validation file {val_file} not found.")
                continue
            except Exception as e:
                print(f"Error loading validation file {val_file}: {e}")
                continue

        # Convert loaded data to TensorFlow tensors
        x_train = tf.convert_to_tensor(
            b_train.reshape([80, dim, dim, dim, 1]),
            dtype=tf.float32
        )
        x_test = tf.convert_to_tensor(
            b_val.reshape([20, dim, dim, dim, 1]),
            dtype=tf.float32
        )

        # Train the model on the current training batch with validation
        hist = model.fit(
            x_train, x_train,
            epochs=epoch_each_iter,
            batch_size=b_size,
            shuffle=True,
            validation_data=(x_test, x_test)
        )

        # Accumulate losses
        training_loss_inner += hist.history['loss']
        validation_loss_inner += hist.history['val_loss']

    # Calculate epoch duration
    time_cg_ml = time.time() - t0

    # Compute average losses using np.mean
    avg_train_loss = np.mean(training_loss_inner)
    avg_val_loss = np.mean(validation_loss_inner)

    # Print the average losses and epoch time
    print(f"Epoch {i} - Training Loss: {avg_train_loss}")
    print(f"Epoch {i} - Validation Loss: {avg_val_loss}")
    print(f"Time for epoch {i}: {time_cg_ml:.2f} seconds.")

    # Append the average losses to the overall loss lists
    training_loss.append(avg_train_loss)
    validation_loss.append(avg_val_loss)

    # Define the directory path for saving the current epoch's model
    save_model_dir = os.path.join(
        project_folder_general,
        project_folder_subname,
        "saved_models",
        f"{project_name}_json_E{epoch_each_iter * i}"
    )
    os.makedirs(save_model_dir, exist_ok=True)

    # Save the model architecture to a JSON file
    model_json = model.to_json()
    model_json_path = os.path.join(save_model_dir, "model.json")
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)

    # Save the model weights to an H5 file
    model_weights_path = os.path.join(save_model_dir, "model.h5")
    model.save_weights(model_weights_path)

    # Save the training and validation loss lists to .npy files
    with open(training_loss_name, 'wb') as f:
        np.save(f, np.array(training_loss))
    with open(validation_loss_name, 'wb') as f:
        np.save(f, np.array(validation_loss))

    # Print the loss histories
    print("Training Loss History:", training_loss)
    print("Validation Loss History:", validation_loss)

# lstsquare_training 0,1 matrixA, normalized b, but 100 train, 10 validation, 1.0e-4
# lstsquareproblem10_training -10, 10 matrix A, normalized b, 100 train, 10 validation, 1.0e-4
# lstsquareproblem100_training -100,100 matrix A, normalized b, 90 train, 10 validation, 1.0e-3
# lstsquareproblem_training 0,1 matrix A, normalized b, 90 train, 10 validation, 1.0e-4
# lstsquareproblem100_training1000 -100,100 matrix A, normalized b, 90 train, 10 validation, 1.0e-3, 1000 epochs

