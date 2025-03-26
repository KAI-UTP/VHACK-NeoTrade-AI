import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

def save_model(model, filename):
    model.save(filename)

def load_model(filename):
    try:
        return keras_load_model(filename)  # ✅ Correctly loads the model using TensorFlow's function
    except Exception as e:
        print(f"Error loading model {filename}: {e}")  # ✅ Debugging message
        return None


