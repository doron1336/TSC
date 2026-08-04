import os.path
import pickle

import numpy as np

from utils.more_features import is_multilabel
from utils.retrieve_minirocket import load_ucr_dataset, apply_minirocket_transform

baseDir = "UCRArchive_2018"
DIRECTORY_NUM = 4
# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), baseDir, item) for item in
            os.listdir(os.path.join(os.getcwd(), baseDir))]

problematic = ['Wafer', 'ECGFiveDays', 'GunPointAgeSpan', 'BeetleFly', 'Adiac', 'ItalyPowerDemand', 'Yoga', 'GunPointOldVersusYoung',
               '.DS_Store', "Missing_value_and_variable_length_datasets_adjusted", 'timer.log']

def save_file(file_path: str, data: np.array):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


for dataset_path in datasets:
    print(dataset_path)
    dataset_name = dataset_path.split("/")[-1]
    if dataset_name == ".DS_Store" or dataset_name == "Missing_value_and_variable_length_datasets_adjusted" or dataset_name in problematic:
        continue
    data_path = os.path.join(f"{dataset_path}_TRAIN.tsv")
    x_train, y_train, x_test, y_test = load_ucr_dataset(data_path)
    if is_multilabel(y_train):  # check if the dataset is relevant - multilable and if we already saved a result for it
        directory = os.path.join(os.getcwd(), baseDir, dataset_name, str(DIRECTORY_NUM))
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
            X_train_transform, X_test_transform = apply_minirocket_transform(x_train, x_test)

            save_file(file_path=filename_train, data=X_train_transform)
            save_file(file_path=filename_test, data=X_test_transform)
            save_file(file_path=filename_y_train, data=y_train)
            save_file(file_path=filename_y_test, data=y_test)
