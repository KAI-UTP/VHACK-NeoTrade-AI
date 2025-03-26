import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

def build_cnn(input_shape):
    """Build and compile a new CNN model"""
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(input_shape[1])  # ✅ Output should match feature size, NOT `input_shape[0] * input_shape[1]`
    ])
    # Compile during creation
    model.compile(optimizer='adam', loss='mse')
    return model

def custom_loss(y_true, y_pred, rewards):
    """Weighted MSE loss that prioritizes features linked to profitable trades."""
    mse = K.mean(K.square(y_true - y_pred), axis=-1)
    weighted_mse = mse * K.abs(rewards)  # Higher reward = More influence
    return K.mean(weighted_mse)

def update_cnn(model, new_data, rewards, epochs=5):
    """Incrementally train CNN with new data while incorporating rewards."""
    if model.optimizer is None:
        model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, rewards))

    # Reshape targets to match the CNN's output shape
    targets = new_data[:, -1, :]  # Take the last time step of each window
    # Shape: (batch_size, num_features) = (116, 14)
    
    # Ensure targets have the same number of samples as new_data
    if len(targets) != len(new_data):
        raise ValueError(f"Data cardinality mismatch. 'x' sizes: {len(new_data)}, 'y' sizes: {len(targets)}")
    
    model.fit(new_data, targets, epochs=epochs, verbose=0)
    return model

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    return tf.keras.models.load_model(filename)