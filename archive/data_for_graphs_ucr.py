import os
import pickle
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import RidgeClassifierCV

from AlgorithmManager import AlgorithmManager
from comaprisons.comparisons_fisher import feature_ranking
from comaprisons.comparisons_relieff import ReliefF
from detach_rocket.detach_classes import DetachRocket
from diffusionMaps.Diffusion_Maps import dm_ranking, dm_ranking_datafold
from models.GA import generations
from utils.JM import JM_flat
from utils.kmeans import perform_kmeans_clustering
from utils.more_features import is_multilabel
from utils.retrieve_minirocket import retrieve_minirocket_data
from utils.topK_indices import calc_score

DIRECTORY_NUM = 4
baseDir = "UCRArchive_2018"
without_GA = 'without_GA'
# chosen_dataset = 'FaceAll'
os.chdir(os.path.join(os.getcwd(), baseDir))
features_array = range(10, 211, 20)


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


failed_datasets = ["MiddlePhalanxTW", "Lightning7", "DistalPhalanxTW", "ACSF1", "PLAID", "Fungi",
                   "LargeKitchenAppliances", "UWaveGestureLibraryX", "ElectricDevices", "DiatomSizeReduction",
                   "ProximalPhalanxTW", "PigAirwayPressure", "PigArtPressure", "Phoneme", "FiftyWords"]
# List the contents of the directory with full paths
# datasets = [os.path.join(os.getcwd(), item) for item in
#             os.listdir(os.path.join(os.getcwd()))]

DetachRocketModel = DetachRocket('minirocket', num_kernels=9996)

# done_datasets_raw = [
#     find_subfolders_with_file(root_folder=dataset, target_subfolder="2", target_file="relief_ga_before") for
#     dataset in datasets]
# done_datasets = list(filter(lambda x: x is not None, done_datasets_raw))

ALGO_NAMES = ["fisher", "mrmr", "relieff", "random", "kmeans_avg_jm"]
miniRocket_results = pd.read_csv('/Users/doron/Desktop/personal/thesis/TSC/datasets_scores_4.csv')
datasets = miniRocket_results.Dataset

for dataset_path in datasets:
    algo_manager = AlgorithmManager(ALGO_NAMES.copy())
    algo_manager.add_algorithm('detached')

    print(dataset_path)
    # done_datasets = done_datasets + failed_datasets
    done_datasets = failed_datasets
    dataset_name = dataset_path.split("/")[-1]
    directory = os.path.join(os.path.abspath("."), dataset_name, str(DIRECTORY_NUM))
    print("directory ", directory)
    if (dataset_name == ".DS_Store" or dataset_name == "Missing_value_and_variable_length_datasets_adjusted"
            or dataset_name == "timer.log" or dataset_name in failed_datasets):
        continue
    check_date = datetime(year=2024, month=12, day=1, hour=12, minute=0, second=0)
    if os.path.exists(os.path.join(directory, 'without_GA', "detach_rocket_data")):
        detach_rocket_data_date = datetime.fromtimestamp(os.path.getmtime(os.path.join(directory, 'without_GA', "detach_rocket_data")))
        if detach_rocket_data_date > check_date:
            continue
    # if os.path.exists(os.path.join(directory, 'selected_210_features')):
    #     continue

    training_data = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_TRAIN.tsv"))
    y_train, x_train = training_data[:, 0].astype(np.int32), training_data[:, 1:]
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
            'dm': dm_ranking,
            'dm_datafold': dm_ranking_datafold,
            'random': random_choosing,
            'kmeans_avg_jm': kmeans_clustering
        }

        try:
            return feature_selection_methods[method](*args)
        except KeyError:
            raise ValueError(f"Unknown feature selection method: {method}")


    # if dataset_name != chosen_dataset:
    #     continue

    os.makedirs(os.path.join(directory, without_GA), exist_ok=True)
    # Load the Data - miniRocket section
    X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)
    # GA section
    if os.path.exists(os.path.join(directory, 'GA_results')):
        with open(os.path.join(directory, 'GA_results'), 'rb') as file:
            GA_results = pickle.load(file)
        chromo_df_bc = GA_results["chromo_df_bc"]
        score_bc = GA_results["score_bc"]
    else:
        chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800,
                                             n_feat=X_train_transform.shape[1],
                                             n_parents=640, mutation_rate=0.20, n_gen=2,
                                             X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train,
                                             Y_test=y_test)
        ga_results = {"chromo_df_bc": chromo_df_bc, "score_bc": score_bc}
        with open(os.path.join(directory, 'GA_results'), 'wb') as file:
            pickle.dump(ga_results, file)

    numOfSelectedFeatures = np.sum(chromo_df_bc, axis=1)
    print("Number of features in each chromo:", numOfSelectedFeatures)
    if os.path.exists(os.path.join(directory, without_GA, 'JM_flat_data_full')):
        with open(os.path.join(directory, without_GA, 'JM_flat_data_full'), 'rb') as file:
            JM_flat_data_full = pickle.load(file)
    else:
        JM_flat_data_full, _ = JM_flat(X_train_transform, y_train)
        with open(os.path.join(directory, without_GA, 'JM_flat_data_full'), 'wb') as file:
            pickle.dump(JM_flat_data_full, file)

    new_X_train_transform = X_train_transform[:, chromo_df_bc[1]]
    JM_flat_data, _ = JM_flat(new_X_train_transform, y_train)
    with open(os.path.join(directory, "JM_flat_data"), 'wb') as file:
        pickle.dump(JM_flat_data, file)

    failed_datasets = []
    start_time = time.time()
    max_index, percentage_vector, sfd_curve, feature_importance_matrix = DetachRocketModel.fit(X=x_train,
                                                                                               y=y_train,
                                                                                               X_test=None,
                                                                                               y_test=None,
                                                                                               X_transfrom=X_train_transform,
                                                                                               X_transform_test=None)
    duration_detached = time.time() - start_time
    detach_rocket_data = {"max_index": max_index, "percentage_vector": percentage_vector, "sfd_curve": sfd_curve,
                          "feature_importance_matrix": feature_importance_matrix}
    detached_predictions = detached_score(detach_rocket_data, X_train_transform, X_test_transform, y_train, y_test)
    algo_manager.set_predictions(algo_name="detached", predictions=detached_predictions)
    algo_manager.add_duration(algo_name='detached', duration=duration_detached)
    with open(os.path.join(directory, without_GA, "detach_rocket_data"), 'wb') as file:
        pickle.dump(detach_rocket_data, file)
    for num_features in features_array:
        # if num_features == 110:
        #     gs = run_pocket(remain_num=num_features, y_train=y_train, X_train_transform=X_train_transform,
        #                     y_test=y_test, X_test_transform=X_test_transform,
        #                     directory=os.path.join(directory, without_GA, "pocket"))
        #     with open(os.path.join(directory, without_GA, f'gs_{num_features}.pickle'), 'wb') as f:
        #         pickle.dump(gs, f)

        tasks = [
            ('fishers', X_train_transform, y_train, num_features),
            ('mrmr', X_train_transform, y_train, num_features),
            ('relieff', X_train_transform, y_train, num_features),
            # ('dm', JM_flat_data, num_features, 100),
            # ('dm_datafold', JM_flat_data, num_features, 100),
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
            with open(os.path.join(directory, f"selected_{num_features}_features"), 'wb') as file:
                pickle.dump(features_dict, file)
        except Exception as e:
            failed_datasets.append(dataset_name)
            with open(os.path.join(directory, f"failed_datasets"), 'wb') as file:
                pickle.dump(failed_datasets, file)
            print(e)
            continue

        # dm_data = {"labels": labels, "dm_coordinates": dm_coordinates}
        # with open(os.path.join(directory, f"dm_data_{num_features}_features"), 'wb') as file:
        #     pickle.dump(dm_data, file)

        for j, name in enumerate(ALGO_NAMES):
            prediction_score = calc_score(X_train_transform[:, features_dict[name]],
                                          X_test_transform[:, features_dict[name]], y_train, y_test)
            algo_manager.add_prediction(algo_name=name, prediction=prediction_score)
            # predictions_dict[algo_dict[j]].append(prediction_score)
        done_datasets.append(dataset_name)
    algo_manager.save(os.path.join(directory, 'algo_manager'))

    # for k, name in enumerate(algo_names):
    #     if name == 'kmeans_avg_jm':
    #         with open(os.path.join(directory, name), 'wb') as file:
    #             pickle.dump(predictions_dict[algo_dict[k]], file)
    #     else:
    #         with open(os.path.join(directory, f'{name}_ga_before'), 'wb') as file:
    #             pickle.dump(predictions_dict[algo_dict[k]], file)
    #     with open(f'{algo_dict[k]}_selected', 'wb') as file:
    #         pickle.dump(algo_dict[k], file)
