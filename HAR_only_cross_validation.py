import os
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import StratifiedKFold

from AlgorithmManager import AlgorithmManager
from comaprisons.comparisons_fisher import feature_ranking
from comaprisons.comparisons_relieff import ReliefF
from detach_rocket.detach_classes import DetachRocket
from utils.JM import JM_flat
from utils.datasets_descriptions import fetch_description
from utils.file_system import save_to_pickle
from utils.kmeans import perform_kmeans_clustering
from utils.more_features import is_multilabel
from utils.retrieve_minirocket import apply_minirocket_transform, load_ucr_dataset, retrieve_minirocket_data
from utils.topK_indices import calc_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.getcwd(), baseDir))
datasets_directory = os.path.join(os.getcwd())
features_array = range(10, 211, 20)
HAR_directory = os.path.join(os.getcwd(), "HAR_datasets")


def check_if_HAR(dataset_name: str) -> bool:
    """
    Check if the dataset is a Human Activity Recognition (HAR) dataset.
    HAR datasets typically have 'HAR' in their name.
    """
    _, type = fetch_description(dataset_name)
    if type == 'HAR':
        return True
    return False


def detached_score(detached_rocket: dict, X_train_transform: np.ndarray, X_test_transform: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray) -> list[float]:
    max_index = detached_rocket["max_index"]
    feature_importance_matrix = detached_rocket["feature_importance_matrix"]
    scores_array = []
    for num_features in features_array:
        classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        detach_rocket_selected = np.argsort(feature_importance_matrix[max_index])[-num_features:]
        classifier_selected.fit(X_train_transform[:, detach_rocket_selected], y_train)
        prediction_score = classifier_selected.score(X_test_transform[:, detach_rocket_selected], y_test)
        print(f"prediction_score for detached rocket model: {prediction_score}")
        scores_array.append(prediction_score)
    return scores_array


DetachRocketModel = DetachRocket('minirocket', num_kernels=9996)

ALGO_NAMES = ["fisher", "mrmr", "relieff", "random", "kmeans_avg_jm"]
miniRocket_results = pd.read_csv('/Users/doron/Desktop/personal/thesis/TSC/datasets_scores_4.csv')
datasets = miniRocket_results.Dataset

for dataset_path in datasets:
    dataset_name = dataset_path.split("/")[-1]
    if not check_if_HAR(dataset_name):
        print(f"Skipping non HAR dataset: {dataset_name}")
        continue
    algo_manager = AlgorithmManager(ALGO_NAMES.copy())
    algo_manager.add_algorithm('detached')

    print(dataset_path)
    directory = os.path.join(HAR_directory, dataset_name)
    os.makedirs(directory, exist_ok=True)
    print("directory ", directory)

    if dataset_name == ".DS_Store" or dataset_name == "Missing_value_and_variable_length_datasets_adjusted" or dataset_name == "timer.log":
        continue

    x_train, y_train, x_test, y_test = load_ucr_dataset(os.path.join(datasets_directory, dataset_path))

    if not is_multilabel(y_train):
        continue


    @algo_manager.time_algorithm_parallel('relieff')
    def relieff_ranking(train, target, num_of_features):
        fs = ReliefF(n_neighbors=110, n_features_to_keep=num_of_features)
        x_train, selected_features = fs.fit_transform(train, np.asarray(target).astype('float'))
        return selected_features


    @algo_manager.time_algorithm_parallel('fisher')
    def fisher_ranking(train, target, num_of_features):
        selector = SelectKBest(score_func=f_classif, k=num_of_features)
        X_new = selector.fit_transform(train, target)
        score = selector.scores_
        return feature_ranking(score, num_of_features)


    @algo_manager.time_algorithm_parallel('mrmr')
    def mrmr_ranking(train, target, num_of_features):
        return mrmr_classif(X=pd.DataFrame(train), y=pd.Series(target), K=num_of_features)


    @algo_manager.time_algorithm_parallel('random')
    def random_choosing(num_of_features):
        array_size = 9996
        random_indices = np.random.choice(
            array_size, num_of_features, replace=False)
        # selected_elements = np.zeros(array_size, dtype=bool)
        # selected_elements[random_indices] = True
        return random_indices


    @algo_manager.time_algorithm_parallel('kmeans_avg_jm')
    def kmeans_clustering(*args):
        return perform_kmeans_clustering(*args)


    def execute_feature_selection(method, *args):
        # Define the mapping inside the function so it can be pickled by joblib
        feature_selection_methods = {
            'fishers': fisher_ranking,
            'mrmr': mrmr_ranking,
            'relieff': relieff_ranking,
            'random': random_choosing,
            'kmeans_avg_jm': kmeans_clustering
        }

        try:
            return feature_selection_methods[method](*args)
        except KeyError:
            raise ValueError(f"Unknown feature selection method: {method}")

    for fold_num, (train_idx, test_idx) in enumerate(cv.split(x_train, y_train), 1):
        os.makedirs(os.path.join(directory, str(fold_num)), exist_ok=True)
        save_path = os.path.join(directory, str(fold_num))
        # Load the Data - miniRocket section
        x_train_fold, x_test_fold = x_train[train_idx], x_train[test_idx]
        y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
        if not os.path.exists(os.path.join(directory, str(fold_num), f"{dataset_name}_minirocket_train")):
            X_train_transform, X_test_transform = apply_minirocket_transform(x_train_fold, x_test_fold)
            save_to_pickle(directory=save_path,
                           file_name=f"{dataset_name}_minirocket_train", data=X_train_transform)
            save_to_pickle(directory=save_path,
                           file_name=f"{dataset_name}_minirocket_test", data=X_test_transform)
        else:
            with open(os.path.join(save_path, f"{dataset_name}_minirocket_train"), 'rb') as file:
                X_train_transform = pickle.load(file)
            with open(os.path.join(save_path, f"{dataset_name}_minirocket_test"), 'rb') as file:
                X_test_transform = pickle.load(file)

        if os.path.exists(os.path.join(save_path, 'JM_flat_data_full')):
            with open(os.path.join(save_path, 'JM_flat_data_full'), 'rb') as file:
                JM_flat_data_full = pickle.load(file)
        else:
            JM_flat_data_full, _ = JM_flat(X_train_transform, y_train_fold)
            with open(os.path.join(save_path, 'JM_flat_data_full'), 'wb') as file:
                pickle.dump(JM_flat_data_full, file)

        failed_datasets = []
        start_time = time.time()
        max_index, percentage_vector, sfd_curve, feature_importance_matrix = DetachRocketModel.fit(X=x_train_fold,
                                                                                                   y=y_train_fold,
                                                                                                   X_test=None,
                                                                                                   y_test=None,
                                                                                                   X_transfrom=X_train_transform,
                                                                                                   X_transform_test=None)
        duration_detached = time.time() - start_time
        detach_rocket_data = {"max_index": max_index, "percentage_vector": percentage_vector, "sfd_curve": sfd_curve,
                              "feature_importance_matrix": feature_importance_matrix}
        detached_predictions = detached_score(detach_rocket_data, X_train_transform, X_test_transform, y_train_fold,
                                              y_test_fold)
        algo_manager.set_predictions(algo_name="detached", predictions=detached_predictions)
        algo_manager.add_duration(algo_name='detached', duration=duration_detached)
        with open(os.path.join(save_path, "detach_rocket_data"), 'wb') as file:
            pickle.dump(detach_rocket_data, file)
        for num_features in features_array:
            tasks = [
                ('fishers', X_train_transform, y_train_fold, num_features),
                ('mrmr', X_train_transform, y_train_fold, num_features),
                ('relieff', X_train_transform, y_train_fold, num_features),
                ('random', num_features),
                ('kmeans_avg_jm', JM_flat_data_full, num_features),
            ]
            try:
                # Parallelize the tasks using joblib
                parallel_results = Parallel(n_jobs=-1)(
                    delayed(execute_feature_selection)(task[0], *task[1:]) for task in tasks)
                algo_manager.collect_parallel_results(parallel_results)

                # Extract results
                features_dict = {name: parallel_results[i]['result'] for i, name in enumerate(ALGO_NAMES)}
                with open(os.path.join(directory, str(fold_num), f"selected_{num_features}_features"), 'wb') as file:
                    pickle.dump(features_dict, file)
            except Exception as e:
                failed_datasets.append(dataset_name)
                with open(os.path.join(directory, f"failed_datasets"), 'wb') as file:
                    pickle.dump(failed_datasets, file)
                print(e)
                continue

            for j, name in enumerate(ALGO_NAMES):
                prediction_score = calc_score(X_train_transform[:, features_dict[name]],
                                              X_test_transform[:, features_dict[name]], y_train_fold, y_test_fold)
                algo_manager.add_prediction(algo_name=name, prediction=prediction_score)
                # predictions_dict[algo_dict[j]].append(prediction_score)
        algo_manager.save(os.path.join(directory, 'algo_manager'))
