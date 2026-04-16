# Here i should go through the results which is saved for every dataset - take those results and find which
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV

from AlgorithmManager import AlgorithmManager
from utils.datasets_descriptions import fetch_description
from utils.file_system import find_subfolders_with_file
from utils.retrieve_minirocket import retrieve_minirocket_data

logging.basicConfig(level=logging.INFO)

baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.getcwd(), baseDir))
# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), item) for item in
            os.listdir(os.path.join(os.getcwd()))]

done_datasets = [
    find_subfolders_with_file(root_folder=dataset, target_subfolder="3", target_file="selected_210_features") for
    dataset in datasets]

features_array = range(10, 211, 20)
DIRECTORY_NUM = 4
columns = ["Algo_name", "Dataset_name", "All_features"] + list(range(10, 211, 20)) + ["Duration", "Type", "Description"]

miniRocket_results = pd.read_csv('/Users/doron/Desktop/personal/thesis/TSC/datasets_scores.csv')


def algo_results_manager(dataset: str, target_subfolder: str, file_name: str) -> pickle:
    return AlgorithmManager.load(os.path.join(dataset, target_subfolder, file_name))


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


def rank_algorithms(df):
    """
    Ranks the algorithms in a DataFrame by row, showing the relative performance of each algorithm.

    Parameters:
    df (pandas.DataFrame): DataFrame with algorithms as columns and prediction scores as rows.

    Returns:
    pandas.DataFrame: Original DataFrame with additional columns showing the rank of each algorithm.
    """
    # Create a copy to avoid modifying the original DataFrame
    ranked_df = df.copy()

    # Apply the rank method to each row
    for i, row in df.iterrows():
        ranked_df.loc[i] = df.loc[i].rank(method='dense', ascending=False)

    return ranked_df


def mask_best_score(dataf: pd):
    # Create a copy of the DataFrame to modify for finding the second-best
    df_temp = dataf.iloc[:, :-2].copy()

    # Replace the best score with a very low value (e.g., -inf)
    for row in features_array:
        df_temp.loc[row, dataf['Best_algo'].loc[row]] = -np.inf
    return df_temp


problematic = ['Wafer', 'GunPointAgeSpan', 'BeetleFly', 'Adiac', 'ItalyPowerDemand', 'Yoga', 'GunPointOldVersusYoung',
               '.DS_Store', "Missing_value_and_variable_length_datasets_adjusted", 'timer.log']
algo_dict = {"fisher": "fisher_ga_before", "mrmr": "mrmr_ga_before", "relief": "relief_ga_before",
             "random": "random_ga_before", "dm": "dm_ga_before", "dm_datafold": "dm_datafold_ga_before"}

data = []
dataframes = {}
for i in datasets:
    dataset_name = i.split('/')[-1]
    try:
        scores_dict = {}
        duration_dict = {}
        directory = os.path.join(os.path.abspath("."), dataset_name, str(DIRECTORY_NUM))
        if not os.path.exists(os.path.join(directory, 'without_GA', "detach_rocket_data")):
            continue
        X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)
        with open(os.path.join(directory, 'without_GA', 'detach_rocket_data'), 'rb') as file:
            detached_rocket_results = pickle.load(file)
        # detached_rocket_scores = detached_score(detached_rocket_results, X_train_transform, X_test_transform,
        #                                         y_train,
        #                                         y_test)
        results = algo_results_manager(dataset=dataset_name, target_subfolder=str(DIRECTORY_NUM),
                                       file_name='algo_manager')
        All_features_score = miniRocket_results.loc[
            miniRocket_results["Dataset"] == dataset_name, "Score_miniRocket"].item()
        for algo_name in results.algorithms:
            scores_dict[algo_name] = results.get_predictions(algo_name)
            if results.get_durations(algo_name) == []:
                duration_dict[algo_name] = [0] * len(features_array)
            else:
                duration_dict[algo_name] = results.get_durations(algo_name)  # Creating a DataFrame with the scores
        for algo in results.algorithms:
            if algo == "detached":
                algo_predictions = results.predictions_dict[algo]
                algo_duration = results.durations_dict[algo][0]
            else:
                algo_predictions = results.predictions_dict[algo]
                algo_duration = results.durations_dict[algo][6]
            dataset_description, dataset_type = fetch_description(dataset_name)
            data.append({"Algo_name": algo,
                         "Dataset_name": dataset_name,
                         "All_features": All_features_score,
                         **{num_features: algo_predictions[index] for index, num_features in enumerate(features_array)},
                         "Duration": algo_duration,
                         "Type": dataset_type,
                         "Description": dataset_description})
    except Exception as e:
        logging.info(e, exc_info=True)

# Create the DataFrame
df = pd.DataFrame(data)

# Reorder columns if needed
df = df[columns]

df.to_csv(os.path.join(os.path.abspath("."), 'scores_dataframe.csv'), index=False)
