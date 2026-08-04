import os
import pickle
import time
from typing import Tuple, List, Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from TSC.AlgorithmManager import AlgorithmManager
from TSC.comaprisons.comparisons_fisher import feature_ranking
from TSC.comaprisons.comparisons_relieff import ReliefF
from detach_rocket.detach_classes import DetachRocket
from TSC.utils.JM import JM_flat
from TSC.utils.datasets_descriptions import fetch_description
from TSC.utils.file_system import save_to_pickle
from TSC.utils.kmeans import perform_kmeans_clustering
from TSC.utils.more_features import is_multilabel
from TSC.utils.retrieve_minirocket import apply_minirocket_transform, load_ucr_dataset
from TSC.utils.topK_indices import calc_score, calc_score_rf
import paths  # noqa: F401  # TSC path config (TSC_CODE_DIR, TSC_DATA_DIR, TSC_RESULTS_DIR)

# HAR_DATASETS = ['GestureMidAirD3', 'GestureMidAirD2', 'UWaveGestureLibraryAll', 'GesturePebbleZ2', 'AllGestureWiimoteX',
#                 'CricketX', 'CricketY']

HAR_DATASETS = ['CricketX']

def check_if_HAR(dataset_name: str) -> bool:
    """
    Check if the dataset is a Human Activity Recognition (HAR) dataset.
    HAR datasets typically have 'HAR' in their name.
    """
    # _, type = fetch_description(dataset_name)
    if dataset_name in HAR_DATASETS:
        return True
    return False


def load_or_compute_minirocket(save_path: str, dataset_name: str, x_train_fold: np.ndarray, x_test_fold: np.ndarray) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Load or compute MiniROCKET transform for train and test data."""
    train_file = os.path.join(save_path, f"{dataset_name}_minirocket_train")
    test_file = os.path.join(save_path, f"{dataset_name}_minirocket_test")

    if os.path.exists(train_file) and os.path.exists(test_file):
        with open(train_file, 'rb') as f:
            X_train_transform = pickle.load(f)
        with open(test_file, 'rb') as f:
            X_test_transform = pickle.load(f)
    else:
        X_train_transform, X_test_transform = apply_minirocket_transform(x_train_fold, x_test_fold)
        save_to_pickle(directory=save_path, file_name=f"{dataset_name}_minirocket_train", data=X_train_transform)
        save_to_pickle(directory=save_path, file_name=f"{dataset_name}_minirocket_test", data=X_test_transform)

    return X_train_transform, X_test_transform


def load_or_compute_jm_flat(save_path: str, X_train_transform: np.ndarray, y_train_fold: np.ndarray) -> np.ndarray:
    """Load or compute JM flat data."""
    jm_file = os.path.join(save_path, 'JM_flat_data_full')
    if os.path.exists(jm_file):
        with open(jm_file, 'rb') as f:
            JM_flat_data_full = pickle.load(f)
    else:
        JM_flat_data_full, _ = JM_flat(X_train_transform, y_train_fold)
        with open(jm_file, 'wb') as f:
            pickle.dump(JM_flat_data_full, f)

    return JM_flat_data_full


def detached_score(detached_rocket: dict, X_train_transform: np.ndarray, X_test_transform: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray) -> list[float]:
    max_index = detached_rocket["max_index"]
    feature_importance_matrix = detached_rocket["feature_importance_matrix"]
    scores_array = []
    for num_features in features_array:
        classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier_selected_2 = RandomForestClassifier(n_estimators=100, random_state=42)
        detach_rocket_selected = np.argsort(feature_importance_matrix[max_index])[-num_features:]
        classifier_selected.fit(X_train_transform[:, detach_rocket_selected], y_train)
        classifier_selected_2.fit(X_train_transform[:, detach_rocket_selected], y_train)
        prediction_score = classifier_selected.score(X_test_transform[:, detach_rocket_selected], y_test)
        prediction_score_2 = classifier_selected_2.score(X_test_transform[:, detach_rocket_selected], y_test)
        print(f"prediction_score for detached rocket model: {prediction_score_2}")
        scores_array.append(prediction_score)
    return scores_array


def setup_feature_selection_methods(algo_manager: AlgorithmManager) -> dict:
    """Define and register feature selection methods with AlgorithmManager."""

    @algo_manager.time_algorithm_parallel('relieff')
    def relieff_ranking(train, target, num_of_features):
        fs = ReliefF(n_neighbors=110, n_features_to_keep=num_of_features)
        _, selected_features = fs.fit_transform(train, np.asarray(target).astype('float'))
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
        random_indices = np.random.choice(array_size, num_of_features, replace=False)
        return random_indices

    @algo_manager.time_algorithm_parallel('kmeans_avg_jm')
    def kmeans_clustering(*args):
        return perform_kmeans_clustering(*args)

    return {
        'fishers': fisher_ranking,
        'mrmr': mrmr_ranking,
        'relieff': relieff_ranking,
        'random': random_choosing,
        'kmeans_avg_jm': kmeans_clustering
    }


def execute_feature_selection(feature_selection_methods: dict, method: str, *args):
    """Execute a feature selection method."""
    try:
        return feature_selection_methods[method](*args)
    except KeyError:
        raise ValueError(f"Unknown feature selection method: {method}")


def process_fold(algo_manager: AlgorithmManager, fold_num: int, train_idx: np.ndarray, test_idx: np.ndarray,
                 X: np.ndarray, y: np.ndarray, dataset_name: str, directory: str, features_array: List[int],
                 feature_selection_methods: dict):
    """Process a single cross-validation fold."""
    save_path = os.path.join(directory, str(fold_num))
    os.makedirs(save_path, exist_ok=True)

    # Split data
    x_train_fold, x_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # Load or compute MiniROCKET transforms
    X_train_transform, X_test_transform = load_or_compute_minirocket(save_path, dataset_name, x_train_fold, x_test_fold)

    # Load or compute JM flat data
    JM_flat_data_full = load_or_compute_jm_flat(save_path, X_train_transform, y_train_fold)

    # Run DetachedRocketModel
    start_time = time.time()
    max_index, percentage_vector, sfd_curve, feature_importance_matrix = DetachRocketModel.fit(
        X=x_train_fold, y=y_train_fold, X_transfrom=X_train_transform
    )
    duration_detached = time.time() - start_time
    detach_rocket_data = {
        "max_index": max_index,
        "percentage_vector": percentage_vector,
        "sfd_curve": sfd_curve,
        "feature_importance_matrix": feature_importance_matrix
    }
    detached_predictions = detached_score(detach_rocket_data, X_train_transform, X_test_transform, y_train_fold,
                                          y_test_fold)
    algo_manager.set_predictions(algo_name="detached", predictions=detached_predictions)
    algo_manager.add_duration(algo_name='detached', duration=duration_detached)
    save_to_pickle(directory=save_path, file_name="detach_rocket_data", data=detach_rocket_data)

    # Feature selection for each number of features
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
                delayed(execute_feature_selection)(feature_selection_methods, task[0], *task[1:]) for task in tasks
            )
            algo_manager.collect_parallel_results(parallel_results)
            features_dict = {name: parallel_results[i]['result'] for i, name in enumerate(ALGO_NAMES)}
            save_to_pickle(directory=save_path, file_name=f"selected_{num_features}_features", data=features_dict)

            # Compute predictions for each algorithm
            for name in ALGO_NAMES:
                prediction_score = calc_score(
                    X_train_transform[:, features_dict[name]],
                    X_test_transform[:, features_dict[name]],
                    y_train_fold,
                    y_test_fold
                )
                prediction_score_rf = calc_score_rf(
                    X_train_transform[:, features_dict[name]],
                    X_test_transform[:, features_dict[name]],
                    y_train_fold,
                    y_test_fold
                )
                algo_manager.add_prediction(algo_name=name, prediction=prediction_score)
        except Exception as e:
            print(f"Error processing {dataset_name} fold {fold_num} with {num_features} features: {e}")
            continue

    # Save AlgorithmManager state
    algo_manager.save(os.path.join(save_path, 'algo_manager'))


DetachRocketModel = DetachRocket('minirocket', num_kernels=9996)

miniRocket_results = pd.read_csv(os.path.join(paths.UCR_DIR, 'datasets_scores_4.csv'))
datasets = miniRocket_results.Dataset


def main(
        datasets: List[str],
        datasets_directory: str,
        HAR_directory: str,
        ALGO_NAMES: List[str],
        features_array: List[int],
        cv: Any
) -> None:
    """Main function to process datasets."""
    failed_datasets: List[str] = []

    for dataset_path in datasets:
        dataset_name = dataset_path.split("/")[-1]
        if not check_if_HAR(dataset_name):
            print(f"Skipping non-HAR dataset: {dataset_name}")
            continue

        if dataset_name in [".DS_Store", "Missing_value_and_variable_length_datasets_adjusted", "timer.log"]:
            continue

        try:
            # Load dataset
            x_train, y_train, x_test, y_test = load_ucr_dataset(os.path.join(datasets_directory, dataset_path))
            X = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)

            if not is_multilabel(y):
                continue

            # Setup directory
            directory = os.path.join(HAR_directory, dataset_name)
            os.makedirs(directory, exist_ok=True)
            print(f"Processing dataset: {dataset_name} at {directory}")

            # Process each fold
            for fold_num, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
                # Initialize AlgorithmManager for each fold
                algo_manager = AlgorithmManager(ALGO_NAMES.copy())
                algo_manager.add_algorithm('detached')

                # Setup feature selection methods
                feature_selection_methods = setup_feature_selection_methods(algo_manager)

                # Process the fold
                process_fold(
                    algo_manager=algo_manager, fold_num=fold_num, train_idx=train_idx, test_idx=test_idx, X=X, y=y,
                    dataset_name=dataset_name, directory=directory, features_array=features_array,
                    feature_selection_methods=feature_selection_methods
                )

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            failed_datasets.append(dataset_name)


if __name__ == "__main__":
    datasets_directory = str(paths.UCR_DIR)
    ALGO_NAMES = ["fisher", "mrmr", "relieff", "random", "kmeans_avg_jm"]

    HAR_directory = os.path.join(paths.UCR_DIR, "HAR_datasets")

    # Example configuration (replace with actual values)
    datasets: List[str] = miniRocket_results.Dataset
    datasets_directory: str = datasets_directory
    HAR_directory: str = HAR_directory
    ALGO_NAMES: List[str] = ALGO_NAMES
    features_array: List[int] = list(range(10, 211, 20))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    main(datasets, datasets_directory, HAR_directory, ALGO_NAMES, features_array, cv)
