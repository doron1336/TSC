import os
import pickle

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import RidgeClassifierCV

from comaprisons.comparisons_fisher import fisher_ranking
from comaprisons.comparisons_mrmr import mrmr_ranking
from comaprisons.comparisons_relieff import ReliefF
from diffusionMaps.Diffusion_Maps import dm_ranking, dm_ranking_datafold
from models.GA import generations
from utils.JM import JM_flat
from utils.more_features import check_multilabel
from utils.timit import record_duration

baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.getcwd(), baseDir))

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


@record_duration
def relieff_ranking(train, target, num_of_features):
    fs = ReliefF(n_neighbors=110, n_features_to_keep=num_of_features)
    x_train, selected_features = fs.fit_transform(train, np.asarray(target).astype('int'))
    return selected_features


def execute_feature_selection(method, *args):
    if method == 'fishers':
        return fisher_ranking(*args)
    elif method == 'mrmr':
        return mrmr_ranking(*args)
    elif method == 'relief':
        return relieff_ranking(*args)
    elif method == 'dm':
        return dm_ranking(*args)
    elif method == 'dm_datafold':
        return dm_ranking_datafold(*args)
    elif method == 'random':
        return random_choosing(*args)


def random_choosing(num_of_features):
    array_size = 9996
    random_indices = np.random.choice(
        array_size, num_of_features, replace=False)
    # selected_elements = np.zeros(array_size, dtype=bool)
    # selected_elements[random_indices] = True
    return random_indices


for dataset_path in datasets:
    print(dataset_path)
    dataset_name = dataset_path.split("/")[-1]
    training_data = np.loadtxt(os.path.join(f"{dataset_path}", f"{dataset_name}_TRAIN.tsv"))
    y_train, x_train = training_data[:, 0].astype(np.int32), training_data[:, 1:]
    if dataset_name in done_datasets and check_multilabel(y_train):
        continue
    for num in range(1, 2):
        directory = os.path.join(os.path.abspath("."), dataset_name, str(num))
        print("directory ", directory)
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
        if os.path.exists(f"{directory}/GA_results"):
            with open(f"{directory}/GA_results", 'rb') as file:
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
        JM_flat_data, dataMean = JM_flat(new_X_train_transform, y_train)
        avg_jm = np.mean(JM_flat_data, axis=1)
        with open(os.path.join(directory, "avg_jm"), 'wb') as file:
            pickle.dump(avg_jm, file)

        algo_dict = {0: "fisher", 1: "mrmr", 2: "relief", 3: "dm", 4: "dm_datafold", 5: "random"}
        predictions_dict = {"fisher": [], "mrmr": [], "relief": [], "dm": [], "dm_datafold": [], "random": []}
        num_algos = len(algo_dict)
        failed_datasets = []
        for num_features in range(10, 201, 10):
            tasks = [
                ('fishers', new_X_train_transform, y_train, num_features),
                ('mrmr', new_X_train_transform, y_train, num_features),
                ('relief', new_X_train_transform, y_train, num_features),
                ('dm', JM_flat_data, num_features, 100),
                ('dm_datafold', JM_flat_data, num_features, 100),
                ('random', num_features)
            ]
            try:
                # Parallelize the tasks using joblib
                results = Parallel(n_jobs=-1)(delayed(execute_feature_selection)(task[0], *task[1:]) for task in tasks)
                # Extract results
                fishers_selected = results[0]
                mrmr_selected = results[1]
                relief_selected = results[2]
                dm_selected, dm_coordinates, labels = results[3]
                dm_selected_datafold, dm_coordinates_datafold = results[4]
                random_selected = results[5]
            except Exception as e:
                failed_datasets.append(dataset_name)
                with open(os.path.join(directory, f"failed_datasets"), 'wb') as file:
                    pickle.dump(failed_datasets, file)
                print(e)
                continue

            features_dict = {0: fishers_selected, 1: mrmr_selected,
                             2: relief_selected, 3: dm_selected, 4: dm_selected_datafold, 5: random_selected}

            dm_data = {"labels": labels, "dm_coordinates": dm_coordinates}
            with open(os.path.join(directory, f"dm_data_{num_features}_features"), 'wb') as file:
                pickle.dump(dm_data, file)
            with open(os.path.join(directory, f"selected_{num_features}_features"), 'wb') as file:
                pickle.dump(features_dict, file)

            for j in range(num_algos):
                classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
                classifier_selected.fit(
                    X_train_transform[:, features_dict[j]], y_train)
                prediction_score = classifier_selected.score(
                    X_test_transform[:, features_dict[j]], y_test)
                predictions_dict[algo_dict[j]].append(prediction_score)

        for k in range(num_algos):
            with open(os.path.join(directory, f'{algo_dict[k]}_ga_before'), 'wb') as file:
                pickle.dump(predictions_dict[algo_dict[k]], file)
            # with open(f'{algo_dict[k]}_selected', 'wb') as file:
            #     pickle.dump(algo_dict[k], file)
