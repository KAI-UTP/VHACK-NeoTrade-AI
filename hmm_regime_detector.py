import numpy as np
from hmmlearn import hmm
import joblib
from sklearn.preprocessing import StandardScaler

def train_hmm(features, n_components=3):
    """Train HMM while ensuring startprob_ is valid."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=100,
        init_params=""  # ✅ Prevents overwriting manually set values
    )

    # ✅ Set `startprob_` to a valid distribution
    model.startprob_ = np.full(n_components, 1 / n_components)  

    # ✅ Set `transmat_` to a uniform probability matrix
    model.transmat_ = np.full((n_components, n_components), 1 / n_components)

    # ✅ Train HMM
    model.fit(scaled_features)

    # ✅ Ensure `startprob_` is valid after training
    if np.isnan(model.startprob_).any() or model.startprob_.sum() == 0:
        model.startprob_ = np.full(n_components, 1 / n_components)

    if np.isnan(model.transmat_).any() or model.transmat_.sum(axis=1).min() == 0:
        model.transmat_ = np.full((n_components, n_components), 1 / n_components)

    return model


def update_hmm(model, features, rewards):
    """Update HMM transition matrix using averaged reward scaling."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    
    model.fit(scaled)  # Retrain HMM with new features

    # ✅ Compute a single reward weight (mean of all rewards)
    reward_weight = np.exp(np.mean(rewards))  # Exponential scaling for numerical stability
    
    # ✅ Scale transition matrix with a single reward value
    model.transmat_ *= reward_weight  # Multiply entire matrix by reward factor
    model.transmat_ /= model.transmat_.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1
    
    return model

def save_hmm(model, filename):
    joblib.dump(model, filename)

def load_hmm(filename):
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        return None