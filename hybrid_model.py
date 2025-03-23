import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


def build_model(input_shape):
    """Build hybrid prediction model"""
    return Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')
    ])

def update_model(old_model, X, y, rewards, epochs=10):
    """Train model using past & current rewards, with normalized weighting."""
    past_rewards = load_rewards()
    all_rewards = np.concatenate([past_rewards, rewards])[-len(X):]  # Only recent rewards

    # Normalize rewards to be between 0 and 1
    if all_rewards.max() > 0:
        all_rewards = (all_rewards - all_rewards.min()) / (all_rewards.max() - all_rewards.min())

    sample_weights = np.abs(all_rewards)
    old_model.fit(X, y, sample_weight=sample_weights, epochs=epochs, verbose=0)

    save_rewards(rewards)  # Persist rewards
    return old_model

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    try:
        return keras_load_model(filename)  # ✅ Correctly loads the model using TensorFlow's function
    except Exception as e:
        print(f"Error loading model {filename}: {e}")  # ✅ Debugging message
        return None

REWARD_FILE = "rewards_history.csv"

def save_rewards(rewards):
    """Save rewards to a CSV file to persist across runs"""
    df = pd.DataFrame({'rewards': rewards})
    if os.path.exists(REWARD_FILE):
        df.to_csv(REWARD_FILE, mode='a', header=False, index=False)  # Append new rewards
    else:
        df.to_csv(REWARD_FILE, index=False)

def load_rewards():
    """Load past rewards from CSV file, handling empty files."""
    if os.path.exists(REWARD_FILE) and os.path.getsize(REWARD_FILE) > 0:
        df = pd.read_csv(REWARD_FILE)
        if 'rewards' in df.columns and not df.empty:
            return df['rewards'].to_numpy()
    return np.array([])  # Return an empty array if no valid rewards exist
