import os
import pickle

import numpy as np

from detach_rocket.detach_classes import DetachRocket
from models.GA import generations
from utils.more_features import is_multilabel
from pocket_wrapper import run_pocket


baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.getcwd(), baseDir))

failed_datasets = ["MiddlePhalanxTW", "Lightning7", "DistalPhalanxTW", "ACSF1", "PLAID", "Fungi",
                   "LargeKitchenAppliances", "UWaveGestureLibraryX", "ElectricDevices", "DiatomSizeReduction",
                   "ProximalPhalanxTW"]

# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), item) for item in
            os.listdir(os.path.join(os.getcwd()))]


def find_subfolders_with_file(root_folder: str, target_subfolder: str, target_file: str) -> str:
    for root, dirs, files in os.walk(root_folder):
        # Check if we're in the target subfolder
        if os.path.basename(root) == target_subfolder:
            if target_file in files:
                # Add the parent folder of the target subfolder to the result list
                return root_folder.split("/")[-1]


done_datasets = [
    find_subfolders_with_file(root_folder=dataset, target_subfolder="1", target_file="selected_200_features") for
    dataset in datasets]

DetachRocketModel = DetachRocket('minirocket', num_kernels=9996)

for dataset_path in datasets:
    print(dataset_path)
    done_datasets = [
        find_subfolders_with_file(root_folder=dataset, target_subfolder="1", target_file="detach_rocket_200") for
        dataset in datasets]
    done_datasets = done_datasets + failed_datasets
    dataset_name = dataset_path.split("/")[-1]
    if dataset_name == ".DS_Store" or dataset_name == "Missing_value_and_variable_length_datasets_adjusted" or dataset_name == "timer.log":
        continue
    training_data = np.loadtxt(os.path.join(f"{dataset_path}", f"{dataset_name}_TRAIN.tsv"))
    y_train, x_train = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    if dataset_name in done_datasets or not is_multilabel(y_train):
        continue
    for num in range(1, 2):
        directory = os.path.join(os.path.abspath("."), dataset_name, str(num))

        # Load the Data - miniRocket section
        with open(f"{directory}/{dataset_name}_minirocket_train", 'rb') as file:
            X_train_transform = pickle.load(file)
        with open(f"{directory}/{dataset_name}_minirocket_test", 'rb') as file:
            X_test_transform = pickle.load(file)
        with open(f"{directory}/{dataset_name}_y_train", 'rb') as file:
            y_train = pickle.load(file)
        with open(f"{directory}/{dataset_name}_y_test", 'rb') as file:
            y_test = pickle.load(file)

        # GA section

        if os.path.exists(os.path.join(directory, 'GA_results')):
            with open(os.path.join(directory, 'GA_results'), 'rb') as file:
                chromo_df_bc, score_bc = pickle.load(file)
        else:
            chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800,
                                                 n_feat=X_train_transform.shape[1],
                                                 n_parents=640, mutation_rate=0.20, n_gen=2,
                                                 X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train,
                                                 Y_test=y_test)
        selectedChromo = np.empty(X_train_transform.shape[1])

        for chromo in chromo_df_bc:
            numOfSelectedFeatures = np.sum(chromo)
            # if numOfSelectedFeatures > max:
            # selectedChromo = chromo
            print("number of features in chromo", numOfSelectedFeatures)
        new_X_train_transform = X_train_transform[:, chromo_df_bc[1]]
        new_X_test_transform = X_test_transform[:, chromo_df_bc[1]]

        for num_features in range(10, 201, 10):

            try:
                run_pocket(remain_num=num_features, y_train=y_train, X_train_transform=X_train_transform,
                           y_test=y_test, X_test_transform=X_test_transform, chromo_df_bc=chromo_df_bc[1],
                           directory=directory)
                max_index, percentage_vector, sfd_curve, feature_importance_matrix = DetachRocketModel.fit(X=X_train_transform,
                                                                                                           y=y_train,
                                                                                                           X_test=None,
                                                                                                           y_test=None,
                                                                                                           X_transfrom=new_X_train_transform,
                                                                                                           X_transform_test=None)
                detach_rocket_data = {"max_index": max_index, "percentage_vector": percentage_vector,
                                      "sfd_curve": sfd_curve,
                                      "feature_importance_matrix": feature_importance_matrix}
                with open(os.path.join(directory, f"detach_rocket_{num_features}"), 'wb') as file:
                    pickle.dump(detach_rocket_data, file)

            except Exception as e:
                failed_datasets.append(dataset_name)
                with open(os.path.join(directory, f"failed_datasets"), 'wb') as file:
                    pickle.dump(failed_datasets, file)
                print(e)
                continue

