# ------------------------------------------------
# 0. Standard imports
# ------------------------------------------------
import os, sys, time, argparse, gc
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.sparse as sparse           # (readable alias)

# ------------------------------------------------
# 1.  Local project imports  (path exactly as before)
# ------------------------------------------------
sys.path.insert(1, '../lib/')
import conjugate_residual as cr         ### CHANGED  (was conjugate_gradient)
import helper_functions as hf

# ------------------------------------------------
# 2.  Command-line interface  (same flags as DCDM)
# ------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--resolution",         type=int, choices=[64, 128], default=64)
parser.add_argument("--total_number_of_epochs",   type=int, default=1000)
parser.add_argument("--epoch_each_number",        type=int, default=1)
parser.add_argument("--batch_size",               type=int, default=10)
parser.add_argument("--loading_number",           type=int, default=100)
parser.add_argument("--gpu_usage",                type=int, default=30,    help="GB")
parser.add_argument("--gpu_idx",                  type=str, default='0')
parser.add_argument("--data_dir",                 type=str,
                    default="/data/hsheng/virtualenvs/minres_improved/2024minresdata/matrixA")
parser.add_argument("--rhs_dir",                  type=str,
                    default="/data/hsheng/virtualenvs/minres_improved/2024minresdata/cr_indef_dataset")
args = parser.parse_args()

# ------------------------------------------------
# 3.  Derived run parameters
# ------------------------------------------------
N               = args.resolution
epoch_num       = args.total_number_of_epochs
epoch_each_iter = args.epoch_each_number
b_size          = args.batch_size
loading_number  = args.loading_number
gpu_usage       = args.gpu_usage
which_gpu       = args.gpu_idx

project_name          = f"3D_N{N}"
project_folder_sub    = os.path.basename(os.getcwd())
project_folder_general= f"../training/3D_N{N}/"

dim2 = N**3
lr    = 1.0e-4

# ------------------------------------------------
# 4.  GPU-memory capping  (same style, fixed to GB→MiB)
# ------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = which_gpu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_usage*1024)])
    except RuntimeError as e:
        print(e)

# ------------------------------------------------
# 5.  Load symmetric-indefinite matrix  A_indefN*.bin
# ------------------------------------------------
A_file_name = os.path.join(args.data_dir, f"A_indefN{N}.bin")  ### CHANGED
A_sparse_scipy = hf.readA_sparse(N, A_file_name, 'f')

# Optional: keep CR handle for later validation
CR = cr.ConjugateResidualSparse(A_sparse_scipy)

# TensorFlow sparse handle once per run
coo     = A_sparse_scipy.tocoo()
indices = np.vstack([coo.row, coo.col]).T
A_sparse = tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)

# ------------------------------------------------
# 6.  Custom CR loss  (keeps original function name for compatibility)
# ------------------------------------------------
def custom_loss_function_cnn_1d_fast(y_true, y_pred):
    """
    Vectorised Conjugate-Residual line-search loss.
    r = y_true, q = y_pred  (both shaped [B, N, N, N, 1])
    L = ‖ r − α A q ‖², α = (rᵀ A q)/(qᵀ A² q)
    """
    eps  = 1e-8
    B    = tf.shape(y_true)[0]
    r    = tf.reshape(y_true, [B, dim2])
    q    = tf.reshape(y_pred, [B, dim2])

    Aq   = tf.sparse.sparse_dense_matmul(A_sparse, tf.transpose(q))
    Aq   = tf.transpose(Aq)                 # shape (B, n)
    AtAq = tf.sparse.sparse_dense_matmul(A_sparse, tf.transpose(Aq))
    AtAq = tf.transpose(AtAq)

    num   = tf.reduce_sum(r * Aq,  axis=1, keepdims=True)  # rᵀ A q
    denom = tf.reduce_sum(q * AtAq, axis=1, keepdims=True) # qᵀ A² q
    denom = tf.where(tf.abs(denom) < eps, tf.sign(denom)*eps, denom)
    alpha = num / denom                                    # (B,1)

    res   = r - alpha * Aq
    return tf.reduce_mean(tf.reduce_sum(tf.square(res), axis=1))

# ------------------------------------------------
# 7.  CNN architecture  (identical to DCDM, comments added)
# ------------------------------------------------
dim     = N
fil_num = 16
input_rhs = keras.Input(shape=(dim, dim, dim, 1))

# --- encoder/decoder with residual blocks (exactly as original) ---
if N == 64:
    # shallow path for 64³
    first = layers.Conv3D(fil_num, 3, padding='same', activation='linear')(input_rhs)
    la    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(first)
    lb    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(la)
    la    = layers.Add()([la, lb])
    lb    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(la)
    # down / up
    apa   = layers.AveragePooling3D(2, padding='same')(lb)
    for _ in range(3):
        apb = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(apa)
        apa = layers.Add()([apa, apb])
    upa   = layers.UpSampling3D(2)(apa)
    upa   = layers.Add()([upa, lb])
    for _ in range(2):
        upb = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(upa)
        upa = layers.Add()([upa, upb])
    last_layer = layers.Dense(1, activation='linear')(upa)

else:  # N == 128
    # deeper path mirrored from the original 128-case
    first = layers.Conv3D(fil_num, 3, padding='same', activation='linear')(input_rhs)
    la    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(first)
    lb    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(la)
    la    = layers.Add()([la, lb])
    lb    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(la)
    la    = layers.Add()([la, lb])
    lb    = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(la)
    apa   = layers.AveragePooling3D(2, padding='same')(lb)
    for _ in range(6):
        apb = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(apa)
        apa = layers.Add()([apa, apb])
    upa   = layers.UpSampling3D(2)(apa)
    upa   = layers.Add()([upa, lb])
    for _ in range(3):
        upb = layers.Conv3D(fil_num, 3, padding='same', activation='relu')(upa)
        upa = layers.Add()([upa, upb])
    last_layer = layers.Dense(1, activation='linear')(upa)

model = keras.Model(input_rhs, last_layer)
model.compile(optimizer="Adam", loss=custom_loss_function_cnn_1d_fast)
model.optimizer.learning_rate.assign(lr)        ### CHANGED (TF-2 API)
model.summary()

# ------------------------------------------------
# 8.  Output logs & checkpoint paths
# ------------------------------------------------
training_loss_name   = os.path.join(project_folder_general, project_folder_sub,
                                    f"{project_name}_training_loss.npy")
validation_loss_name = os.path.join(project_folder_general, project_folder_sub,
                                    f"{project_name}_validation_loss.npy")
training_loss, validation_loss = [], []

# ------------------------------------------------
# 9.  RHS dataset folder (now points to CR-indef set)
# ------------------------------------------------
foldername = args.rhs_dir.rstrip("/") + "/"

total_data_points = len(os.listdir(foldername))   ### NEW  (auto detect)
for_loading_number = total_data_points // loading_number
b_rhs = np.zeros([loading_number, dim2], dtype=np.float32)

# ------------------------------------------------
# 10.  Epoch loop  (structure unchanged)
# ------------------------------------------------
for epoch in range(epoch_num):
    print(f"\n=== Epoch {epoch+1}/{epoch_num} ===")
    perm = np.random.permutation(total_data_points)
    train_losses, val_losses = [], []
    t0 = time.time()

    for blk in range(for_loading_number):
        # -------- load mini-batch .npy vectors --------
        for j in range(loading_number):
            idx = perm[blk*loading_number + j]
            b_rhs[j] = np.load(os.path.join(foldername, f"b_{idx}.npy"))

        split = int(0.9 * loading_number)
        x_train = tf.convert_to_tensor(b_rhs[:split].reshape([split, N, N, N, 1]),
                                       dtype=tf.float32)
        x_val   = tf.convert_to_tensor(b_rhs[split:].reshape([loading_number-split,
                                                              N, N, N, 1]),
                                       dtype=tf.float32)

        hist = model.fit(x_train, x_train,
                         validation_data=(x_val, x_val),
                         epochs=epoch_each_iter,
                         batch_size=b_size,
                         shuffle=True,
                         verbose=0)

        train_losses.extend(hist.history['loss'])
        val_losses.extend(hist.history['val_loss'])

    # -------- epoch summary --------
    print("  training loss =", np.mean(train_losses),
          " validation loss =", np.mean(val_losses),
          " elapsed =", time.time()-t0, "s")
    training_loss.append(np.mean(train_losses))
    validation_loss.append(np.mean(val_losses))

    # -------- checkpoint --------
    save_dir = os.path.join(project_folder_general, project_folder_sub,
                            "saved_models", f"{project_name}_json_E{epoch+1}")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "model.json"), "w") as f:
        f.write(model.to_json())
    model.save_weights(os.path.join(save_dir, "model.h5"))

    # -------- persist loss curves --------
    np.save(training_loss_name,   np.array(training_loss))
    np.save(validation_loss_name, np.array(validation_loss))
    gc.collect()
