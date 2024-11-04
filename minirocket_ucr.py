import os.path
import pickle

import numpy as np

from models.minirocket import fit, transform
from utils.more_features import is_multilabel

baseDir = "UCRArchive_2018"
# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), baseDir, item) for item in
            os.listdir(os.path.join(os.getcwd(), baseDir))]


def save_file(file_path: str, data: np.array):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


for dataset_path in datasets:
    print(dataset_path)
    dataset_name = dataset_path.split("/")[-1]
    if dataset_name == ".DS_Store" or dataset_name == "Missing_value_and_variable_length_datasets_adjusted" or dataset_name == "timer.log":
        continue
    data_path = os.path.join(f"{dataset_path}_TRAIN.tsv")

    training_data = np.loadtxt(os.path.join(f"{dataset_path}", f"{dataset_name}_TRAIN.tsv"))
    y_train, x_train = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    x_train[np.isnan(x_train)] = 0  # fill missing data with 0
    if is_multilabel(y_train):  # check if the dataset is relevant - multilable and if we already saved a result for it
        test_data = np.loadtxt(os.path.join(f"{dataset_path}", f"{dataset_name}_TEST.tsv"))
        y_test, x_test = test_data[:, 0].astype(np.int32), test_data[:, 1:]
        x_test[np.isnan(x_test)] = 0  # fill missing data with 0

        for num in range(1, 4):
            directory = os.path.join(os.getcwd(), baseDir, dataset_name, str(num))
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory)
                except:
                    print("failed to create directory")

            # miniRocket section
            filename_train = os.path.join(os.path.abspath("."), directory, f"{dataset_name}_minirocket_train")
            filename_test = os.path.join(os.path.abspath("."), directory, f"{dataset_name}_minirocket_test")
            filename_y_train = os.path.join(os.path.abspath("."), directory, f"{dataset_name}_y_train")
            filename_y_test = os.path.join(os.path.abspath("."), directory, f"{dataset_name}_y_test")
            if not os.path.exists(filename_train):
                parameters = fit(x_train)
                X_train_transform = transform(x_train, parameters)
                X_test_transform = transform(x_test, parameters)

                save_file(file_path=filename_train, data=X_train_transform)
                save_file(file_path=filename_test, data=X_test_transform)
                save_file(file_path=filename_y_train, data=y_train)
                save_file(file_path=filename_y_test, data=y_test)
