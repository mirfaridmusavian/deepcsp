import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def kfold_by_run(train_metadata: pd.DataFrame, run_ids: np.ndarray):
    
    train_indices = (train_metadata[train_metadata['run'] != run_id].index.values for run_id in run_ids)
    valid_indices = (train_metadata[train_metadata['run'] == run_id].index.values for run_id in run_ids)
    return zip(train_indices, valid_indices)

def kfold_by_session(train_metadata: pd.DataFrame, session_ids: np.ndarray):
    
    train_indices = (train_metadata[train_metadata['session'] != session_id].index.values for session_id in session_ids)
    valid_indices = (train_metadata[train_metadata['session'] == session_id].index.values for session_id in session_ids)
    return zip(train_indices, valid_indices)

def partition_BNCI2014001(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """
    This dataset has 2 sessions and 5 runs each
    second session for test
    and kfold the first session by runs
    """
    train_metadata = metadata[metadata['session'] == 'session_T']
    run_ids = train_metadata.run.unique()
    kfold = kfold_by_run(train_metadata, run_ids)
    X_ = X[train_metadata.index.values]
    y_ = y[train_metadata.index.values]
    X_test = X[metadata['session'] == 'session_E']
    y_test = y[metadata['session'] == 'session_E']

    return X_, X_test, y_, y_test, kfold

def partition_BNCI2014002(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):

    """
    This dataset has 8 runs and 1 session:
    Reserve last 2 runs for test
    make 6 fold cv with the rest
    """

    test_ind = (metadata['run'] == 'run_7') | (metadata['run'] == 'run_6')
    run_ids = metadata.run.unique()[:-2]
    train_ind = ~test_ind
    train_metadata = metadata[train_ind]
    kfold = kfold_by_run(train_metadata, run_ids)

    X_ = X[train_metadata.index.values]
    y_ = y[train_metadata.index.values]
    X_test = X[test_ind]
    y_test = y[test_ind]

    return X_, X_test, y_, y_test, kfold

def partition_BNCI2014004(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):

    """
    This dataset has 5 sessions:
    Reserve last run for test
    make 4 fold cv with the rest
    """

    train_metadata = metadata[metadata['session'] != 'session_4']
    session_ids = train_metadata.session.unique()
    kfold = kfold_by_session(train_metadata, session_ids)
    X_ = X[train_metadata.index.values]
    y_ = y[train_metadata.index.values]
    X_test = X[metadata['session'] == 'session_4']
    y_test = y[metadata['session'] == 'session_4']

    return X_, X_test, y_, y_test, kfold


def partition_BNCI2015001(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """
    This dataset has 2 sessions and 1 run each
    2ed session for test
    random kfold the other one
    """
    X_ = X[metadata['session'] == 'session_A']
    y_ = y[metadata['session'] == 'session_A']
    X_test = X[metadata['session'] == 'session_B']
    y_test = y[metadata['session'] == 'session_B']
    skf = StratifiedKFold(n_splits=5)
    kfold = skf.split(X_, y_)
    return X_, X_test, y_, y_test, kfold

def partition_BNCI2015004(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """
    This dataset has 2 sessions and 1 run each
    2ed session for test
    random kfold the other one
    """
    X_ = X[metadata['session'] == 'session_0']
    y_ = y[metadata['session'] == 'session_0']
    X_test = X[metadata['session'] == 'session_1']
    y_test = y[metadata['session'] == 'session_1']
    skf = StratifiedKFold(n_splits=5)
    kfold = skf.split(X_, y_)
    return X_, X_test, y_, y_test, kfold

def partition_Cho2017(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """
    This dataset has 1 session and 1 run and 200 trials
    reserve last 40 trials for test
    random kfold the others
    """
    X_ = X[:160]
    y_ = y[:160]
    X_test = X[160:]
    y_test = y[160:]
    skf = StratifiedKFold(n_splits=5)
    kfold = skf.split(X_, y_)
    return X_, X_test, y_, y_test, kfold

def partition_MunichMI(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """
    This dataset has 1 session and 1 run and 300 trials
    reserve last 50 trials for test
    random kfold the others
    """
    X_ = X[:250]
    y_ = y[:250]
    X_test = X[250:]
    y_test = y[250:]
    skf = StratifiedKFold(n_splits=5)
    kfold = skf.split(X_, y_)
    return X_, X_test, y_, y_test, kfold


# modify the folloing in your installed enviroment
# BASE_URL = 'https://physionet.org/files/eegmmidb/1.0.0/'
# find it according to the error log)

def partition_PhysionetMI(X: np.ndarray, y: np.ndarray, metadata: pd.DataFrame):
    """
    This dataset has 1 session and 1 run and 45 trials
    reserve last 15 trials for test
    random kfold the others
    """
    X_ = X[:30]
    y_ = y[:30]
    X_test = X[30:]
    y_test = y[30:]
    skf = StratifiedKFold(n_splits=3)
    kfold = skf.split(X_, y_)
    return X_, X_test, y_, y_test, kfold





