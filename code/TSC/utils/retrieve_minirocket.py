import os
import pickle

import numpy as np

from TSC.models.minirocket import fit, transform


def retrieve_minirocket_data(directory: str, dataset_name: str):
    with open(os.path.join(directory, f"{dataset_name}_minirocket_train"), 'rb') as file:
        x_train_transform = pickle.load(file)
    with open(os.path.join(directory, f"{dataset_name}_minirocket_test"), 'rb') as file:
        x_test_transform = pickle.load(file)
    with open(os.path.join(directory, f"{dataset_name}_y_train"), 'rb') as file:
        y_train = pickle.load(file)
    with open(os.path.join(directory, f"{dataset_name}_y_test"), 'rb') as file:
        y_test = pickle.load(file)
    return x_train_transform, x_test_transform, y_train, y_test


def apply_minirocket_transform(x_train, x_test):
    parameters = fit(x_train)
    X_train_transform = transform(x_train, parameters)
    X_test_transform = transform(x_test, parameters)
    return X_train_transform, X_test_transform

def load_ucr_dataset(dataset_path: str):
    dataset_name = dataset_path.split("/")[-1]
    training_data = np.loadtxt(os.path.join(f"{dataset_path}", f"{dataset_name}_TRAIN.tsv"))
    test_data = np.loadtxt(os.path.join(f"{dataset_path}", f"{dataset_name}_TEST.tsv"))
    y_train, x_train = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    x_train[np.isnan(x_train)] = 0  # fill missing data with 0
    y_test, x_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]
    x_test[np.isnan(x_test)] = 0  # fill missing data with 0
    return x_train, y_train, x_test, y_test
