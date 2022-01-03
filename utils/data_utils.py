import numpy as np
import pandas as pd

def partition_BNCI2015001(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    X_ = X[metadata['session'] == 'session_A']
    y_ = y[metadata['session'] == 'session_A']
    X_test = X[metadata['session'] == 'session_B']
    y_test = y[metadata['session'] == 'session_B']

    return X_, X_test, y_, y_test




