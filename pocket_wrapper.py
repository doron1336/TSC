# from Srocket_main import Srocket
import os
import pickle
from typing import List

import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from POCKET.MiniROCKET.PROCKET_pruner import PROCKETPruner


pruning_remain_rates = [0.10, 0.35, 0.10, 0.30, 0.19, 0.49, 0.01, 0.39, 0.30, 0.33, 0.34, 0.19, 0.20, 0.62, 0.54, 0.27,
                        0.33, 0.73, 0.40, 0.01, 0.01, 0.20, 0.39, 0.21, 0.29, 0.37, 0.72, 0.67, 0.80, 0.78]

pruning_remain_numbers = [int(10000 * item) for item in
                          [0.10, 0.35, 0.10, 0.30, 0.19, 0.49, 0.01, 0.39, 0.30, 0.33, 0.34, 0.19, 0.20, 0.62, 0.54,
                           0.27, 0.33, 0.73, 0.40, 0.01, 0.01, 0.20, 0.39, 0.21, 0.29, 0.37, 0.72, 0.67, 0.80,
                           0.78]]


def score(y_true, y_predict):
    acc = np.mean(y_predict == y_true)

    return acc


print(pruning_remain_numbers)
my_scorer = make_scorer(score, greater_is_better=True)
# remain_num = pruning_remain_numbers[0]

remain_num = 100


def run_pocket(remain_num: int, y_train: List[str], X_train_transform: np.ndarray, y_test: List[str],
               X_test_transform: np.ndarray, chromo_df_bc: List[bool], directory: str):
    stop_thr = 0.0001
    num_epochs = 50

    y_train = np.asarray([int(x) for x in y_train])
    y_test = np.asarray([int(x) for x in y_test])

    new_X_train_transform = X_train_transform[:, chromo_df_bc]
    new_X_test_transform = X_test_transform[:, chromo_df_bc]

    X_training_transform_copy = new_X_train_transform.copy()
    Y_training_copy = y_train.copy()
    X_test_transform_copy = new_X_test_transform.copy()
    Y_test_copy = y_test.copy()

    # data normalization
    mean = np.mean(new_X_train_transform, axis=0)

    new_X_train_transform -= mean
    norm = np.linalg.norm(new_X_train_transform, axis=0)
    norm[norm == 0] = 1
    new_X_train_transform /= norm

    new_X_test_transform -= mean
    new_X_test_transform /= norm

    # Y +-1 coded
    n_class = int(np.max(np.asarray(y_train))) + 1
    Y_training_coded = np.ones([n_class, n_class]) * -1
    for _ in range(Y_training_coded.shape[0]):
        Y_training_coded[_, _] = 1
    Y_training_coded = Y_training_coded[y_train]

    gs = GridSearchCV(
        PROCKETPruner(n_class, y_train, new_X_test_transform, y_test, remain_num=remain_num,
                      stop_thr=stop_thr, epoch=num_epochs),
        {'k': [10, 1, 0.1] + [0.01, 100, 1000]}, cv=5, n_jobs=-1, return_train_score=True, scoring=my_scorer)

    gs.fit(new_X_train_transform, y_train.copy())
    with open(os.path.join(directory, f'gs_{remain_num}.pickle'), 'wb') as f:
        pickle.dump(gs, f)
    estimator = gs.best_estimator_

    '''post training'''
    retrain_time, retrain_scores_thr_based, retrain_alpha_thr_based = \
        estimator.retrain(X_training_transform_copy, Y_training_copy, X_test_transform_copy, Y_test_copy,
                          model='PPV')
