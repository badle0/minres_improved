import numpy as np
import matplotlib.pyplot as plt

# File paths for training and validation loss data
training_loss_file = '/data/hsheng/virtualenvs/minres_improved/src/train/lstsquares_filtered_training_dcdmloss/3D_N64/train/3D_N64_training_loss.npy'
validation_loss_file = '/data/hsheng/virtualenvs/minres_improved/src/train/lstsquares_filtered_training_dcdmloss/3D_N64/train/3D_N64_validation_loss.npy'
# Load the loss data from the .npy files
try:
    training_loss = np.load(training_loss_file)
    print(f"Training loss data loaded: {training_loss[:15]}")  # Print first 100 values for verification
except Exception as e:
    print(f"Error loading training loss data: {e}")
    training_loss = None

try:
    validation_loss = np.load(validation_loss_file)
    print(f"Validation loss data loaded: {validation_loss[:15]}")  # Print first 100 values for verification
except Exception as e:
    print(f"Error loading validation loss data: {e}")
    validation_loss = None

# Determine the best epoch based on the lowest validation loss
if validation_loss is not None and len(validation_loss) > 0:
    best_epoch = np.argmin(validation_loss)
    best_val_loss = validation_loss[best_epoch]
    print(f"The best epoch is {best_epoch+1} with a validation loss of {best_val_loss:.8f}")
else:
    print("No validation loss data available to determine the best epoch.")

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
if training_loss is not None:
    plt.plot(training_loss, label='Training Loss')
else:
    print("No training loss data to plot.")

if validation_loss is not None:
    plt.plot(validation_loss, label='Validation Loss')
else:
    print("No validation loss data to plot.")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

# Zoom into the y-axis to visualize smaller differences
plt.ylim(0.07, 0.01)

plt.legend()
plt.grid(True)
plt.show()
