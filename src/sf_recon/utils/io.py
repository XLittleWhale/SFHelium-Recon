import numpy as np
import os

def save_npz(path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **kwargs)

def load_npz(path):
    return np.load(path)
