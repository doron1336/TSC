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

from AlgorithmManager import AlgorithmManager
from comaprisons.comparisons_fisher import feature_ranking
from comaprisons.comparisons_relieff import ReliefF
from detach_rocket.detach_classes import DetachRocket
from models.GA import generations
from utils.JM import JM_flat
from utils.kmeans import perform_kmeans_clustering
from utils.more_features import is_multilabel
from utils.retrieve_minirocket import retrieve_minirocket_data
from utils.topK_indices import calc_score

DIRECTORY_NUM = 4
baseDir = "UCRArchive_2018"
without_GA = 'without_GA'
dataset_name = 'CricketX'

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


DetachRocketModel = DetachRocket('minirocket', num_kernels=9996)

ALGO_NAMES = ["fisher", "mrmr", "relieff", "random", "kmeans_avg_jm"]

algo_manager = AlgorithmManager(ALGO_NAMES.copy())
algo_manager.add_algorithm('detached')

print(f"Processing dataset: {dataset_name}")

dataset_path = os.path.join(os.path.abspath("."), dataset_name)
directory = os.path.join(dataset_path, str(DIRECTORY_NUM))
print(f"Directory: {directory}")

# Create directory if it doesn't exist
os.makedirs(os.path.join(directory, without_GA), exist_ok=True)

# Load training data
training_data = np.loadtxt(os.path.join(dataset_path, f"{dataset_name}_TRAIN.tsv"))
y_train, x_train = training_data[:, 0].astype(np.int32), training_data[:, 1:]

if not is_multilabel(y_train):
    print(f"Dataset {dataset_name} is not multilabel, skipping")
    exit(1)


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
    return random_indices


@algo_manager.time_algorithm_parallel('kmeans_avg_jm')
def kmeans_clustering(*args):
    return perform_kmeans_clustering(*args)


def execute_feature_selection(method, *args):
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


# Load the Data - miniRocket section
print("Loading MiniRocket data...")
X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)

# GA section
print("Running GA...")
if os.path.exists(os.path.join(directory, 'GA_results')):
    with open(os.path.join(directory, 'GA_results'), 'rb') as file:
        GA_results = pickle.load(file)
    chromo_df_bc = GA_results["chromo_df_bc"]
    score_bc = GA_results["score_bc"]
    print("Loaded existing GA results")
else:
    chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800,
                                         n_feat=X_train_transform.shape[1],
                                         n_parents=640, mutation_rate=0.20, n_gen=2,
                                         X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train,
                                         Y_test=y_test)
    ga_results = {"chromo_df_bc": chromo_df_bc, "score_bc": score_bc}
    with open(os.path.join(directory, 'GA_results'), 'wb') as file:
        pickle.dump(ga_results, file)
    print("GA results saved")

numOfSelectedFeatures = np.sum(chromo_df_bc, axis=1)
print("Number of features in each chromo:", numOfSelectedFeatures)

# JM flat data
print("Computing JM flat data...")
if os.path.exists(os.path.join(directory, without_GA, 'JM_flat_data_full')):
    with open(os.path.join(directory, without_GA, 'JM_flat_data_full'), 'rb') as file:
        JM_flat_data_full = pickle.load(file)
    print("Loaded existing JM flat data full")
else:
    JM_flat_data_full, _ = JM_flat(X_train_transform, y_train)
    with open(os.path.join(directory, without_GA, 'JM_flat_data_full'), 'wb') as file:
        pickle.dump(JM_flat_data_full, file)
    print("JM flat data full saved")

new_X_train_transform = X_train_transform[:, chromo_df_bc[1]]
JM_flat_data, _ = JM_flat(new_X_train_transform, y_train)
with open(os.path.join(directory, "JM_flat_data"), 'wb') as file:
    pickle.dump(JM_flat_data, file)
print("JM flat data saved")

# Detached Rocket
print("Running Detached Rocket...")
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
print(f"Detached Rocket completed in {duration_detached:.2f} seconds")

# Feature selection and scoring
print("Running feature selection algorithms...")
for num_features in features_array:
    print(f"\nProcessing {num_features} features...")

    tasks = [
        ('fishers', X_train_transform, y_train, num_features),
        ('mrmr', X_train_transform, y_train, num_features),
        ('relieff', X_train_transform, y_train, num_features),
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
        print(f"Selected features saved for {num_features}")
    except Exception as e:
        print(f"Error during feature selection for {num_features} features: {e}")
        continue

    for j, name in enumerate(ALGO_NAMES):
        prediction_score = calc_score(X_train_transform[:, features_dict[name]],
                                      X_test_transform[:, features_dict[name]], y_train, y_test)
        algo_manager.add_prediction(algo_name=name, prediction=prediction_score)
        print(f"{name}: {prediction_score:.4f}")

# Save algorithm manager
algo_manager.save(os.path.join(directory, 'algo_manager'))
print(f"\nAlgorithm manager saved to {os.path.join(directory, 'algo_manager')}")
print("CricketX processing completed successfully!")