import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from AlgorithmManager import AlgorithmManager
from utils.retrieve_minirocket import retrieve_minirocket_data

features_array = range(10, 211, 20)
DIRECTORY_NUM = 4


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


baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.getcwd(), baseDir))
# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), item) for item in
            os.listdir(os.path.join(os.getcwd()))]
dataset_name = 'GestureMidAirD3'
data = {}
try:
    scores_dict = {}
    duration_dict = {}
    directory = os.path.join(os.path.abspath("."), dataset_name, str(DIRECTORY_NUM))

    X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)
    with open(os.path.join(directory, 'without_GA', 'detach_rocket_data'), 'rb') as file:
        detached_rocket_results = pickle.load(file)
    # detached_rocket_scores = detached_score(detached_rocket_results, X_train_transform, X_test_transform,
    #                                         y_train,
    #                                         y_test)
    results = algo_results_manager(dataset=dataset_name, target_subfolder=str(DIRECTORY_NUM),
                                   file_name='algo_manager')
    # All_features_score = miniRocket_results.loc[
    #     miniRocket_results["Dataset"] == dataset_name, "Score_miniRocket"].item()
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
        # Classification scores for 5 algorithms (y-axis)
        data[algo] = [algo_predictions[index] for index in range(len(features_array))]

except Exception as e:
    print(e)




# Create the plot
plt.figure(figsize=(10, 6))

# Plot each algorithm's performance with distinct colors
plt.plot(features_array, data["fisher"], marker='o', color='#1f77b4', label='Fisher')         # Blue
plt.plot(features_array, data["mrmr"], marker='s', color='#ff7f0e', label='MrMr')            # Orange
plt.plot(features_array, data["relieff"], marker='^', color='#2ca02c', label='Relieff')      # Green
plt.plot(features_array, data['random'], marker='*', color='#9467bd', label='Random')        # Purple
plt.plot(features_array, data["kmeans_avg_jm"], marker='d', color='#000000', label='Kmeans_ang_JM')  # Black
plt.plot(features_array, data["detached"], marker='x', color='#17becf', label='Detached')    # Cyan

# Customize the plot
plt.xlabel('Number of Features Selected')
plt.ylabel('Classification Score')
plt.title('Classification Score vs. Number of Features Selected')
plt.grid(True)
plt.legend()  # Display the legend
plt.xticks(features_array)  # Ensure all feature numbers are shown on x-axis

# Display the plot
plt.tight_layout()
plt.show()
