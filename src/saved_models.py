from tensorflow import keras
import tensorflow as tf
import os

model_dir = "/data/hsheng/virtualenvs/minres_improved/lstsquares_filtered_training/3D_N64/minres_improved/saved_models/3D_N64_json_E101"  # the folder containing model.json and model.h5

# Load model architecture from JSON.
with open(os.path.join(model_dir, 'model.json'), 'r') as f:
    model_json = f.read()
model = keras.models.model_from_json(model_json)

# Load weights.
model.load_weights(os.path.join(model_dir, 'model.h5'))

saved_model_dir = os.path.join(model_dir, 'saved_model')
tf.saved_model.save(model, saved_model_dir)

