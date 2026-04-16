# Here i should go through the results which is saved for every dataset - take those results and find which
import logging
import os
import pickle

import pandas as pd

from AlgorithmManager import AlgorithmManager
from utils.datasets_descriptions import fetch_description

logging.basicConfig(level=logging.INFO)
path = '/Users/doron/Desktop/personal/thesis/TSC/UCRArchive_2018'
os.chdir(path)
HAR_directory = os.path.join(os.getcwd(), "HAR_datasets")

# List the contents of the directory with full paths
datasets = ["GestureMidAirD3", "GestureMidAirD2", "GesturePebbleZ2", "UWaveGestureLibraryAll",
            "AllGestureWiimoteX", "CricketX", "CricketY"]

FEATURES_ARRAY = range(10, 211, 20)
columns = ["Algo_name", "Dataset_name", "All_features"] + list(range(10, 211, 20)) + ["Type"]

miniRocket_results = pd.read_csv('/Users/doron/Desktop/personal/thesis/TSC/datasets_scores.csv')


def algo_results_manager(dataset_path: str, target_subfolder: str, file_name: str) -> pickle:
    return AlgorithmManager.load(os.path.join(dataset_path, target_subfolder, file_name))


def aggregate_scores(algo_name: str, dataset_path: str) -> pd.DataFrame:
    """
    Aggregates the scores for each algorithm across all datasets.
    """
    scores_dict = {}
    for i in range(1, 6):
        results = algo_results_manager(dataset_path=dataset_path, target_subfolder=str(i),
                                       file_name='algo_manager')
        scores_dict[i] = results.get_predictions(algo_name)
    df = pd.DataFrame.from_dict(scores_dict).T
    means = df.mean()
    stds = df.std()
    new_df = pd.DataFrame({
        col: [f"{means[col]:.4f} ± {stds[col]:.4f}"] for col in df.columns
    })

    return new_df


def aggregate_durations(algo_name: str, dataset_path: str) -> pd.DataFrame:
    """
    Aggregates the scores for each algorithm across all datasets.
    """
    duration_dict = {}
    for i in range(1, 6):
        results = algo_results_manager(dataset_path=dataset_path, target_subfolder=str(i),
                                       file_name='algo_manager')
        if algo_name == "detached":
            duration_dict[i] = [results.get_durations(algo_name)[0]] * len(FEATURES_ARRAY)
        else:
            duration_dict[i] = results.get_durations(algo_name)
    df = pd.DataFrame.from_dict(duration_dict).T
    means = df.mean()
    stds = df.std()
    new_df = pd.DataFrame({
        col: [f"{means[col]:.4f} ± {stds[col]:.4f}"] for col in df.columns
    })

    return new_df


algo_dict = {"fisher": "fisher_ga_before", "mrmr": "mrmr_ga_before", "relief": "relief_ga_before",
             "random": "random_ga_before", "dm": "dm_ga_before", "dm_datafold": "dm_datafold_ga_before"}

data = []
dataframes = {}
for dataset_name in datasets:
    try:
        duration_dict = {}
        dataset_directory = os.path.join(HAR_directory, dataset_name)
        results = algo_results_manager(dataset_path=dataset_directory, target_subfolder='1',
                                       file_name='algo_manager')
        All_features_score = miniRocket_results.loc[
            miniRocket_results["Dataset"] == dataset_name, "Score_miniRocket"].item()
        for algo_name in results.algorithms:
            scores_df = aggregate_scores(algo_name=algo_name, dataset_path=dataset_directory)
            durations_df = aggregate_durations(algo_name=algo_name, dataset_path=dataset_directory)
            dataset_description, dataset_type = fetch_description(dataset_name)
            data.append({"Algo_name": algo_name,
                         "Dataset_name": dataset_name,
                         "All_features": All_features_score,
                         **{num_features: f"Score: {scores_df.iloc[0, index]} Duration: {durations_df.iloc[0, index]}"
                            for index, num_features in enumerate(FEATURES_ARRAY)},
                         "Type": dataset_type})
    except Exception as e:
        logging.info(e, exc_info=True)

    # Create the DataFrame
df = pd.DataFrame(data)

# Reorder columns if needed
df = df[columns]

df.to_csv(os.path.join(HAR_directory, 'scores_dataframe_fold2.csv'), index=False)
