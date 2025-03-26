import numpy as np
from hmmlearn import hmm
import joblib
from sklearn.preprocessing import StandardScaler

def save_hmm(model, filename):
    joblib.dump(model, filename)

def load_hmm(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        return None