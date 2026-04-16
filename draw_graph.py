import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from AlgorithmManager import AlgorithmManager
from utils.retrieve_minirocket import retrieve_minirocket_data

features_array = range(10, 211, 20)
# DIRECTORY_NUM will be set in the loop for each fold
NUM_FOLDS = 5  # Adjust this to match your actual number of folds


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


baseDir = "UCRArchive_2018/HAR_datasets"
os.chdir(os.path.join(os.getcwd(), baseDir))

# dataset_name = 'GestureMidAirD3'
HAR_DATASETS = ['GestureMidAirD3', 'GestureMidAirD2', 'UWaveGestureLibraryAll', 'GesturePebbleZ2', 'AllGestureWiimoteX',
                'CricketX', 'CricketY']
# HAR_DATASETS = ['GestureMidAirD3']

# Create main output directory
output_base_dir = '/Users/doron/Desktop/personal/thesis/TSC/plots_averaged'
os.makedirs(output_base_dir, exist_ok=True)

# Loop through each dataset
for dataset_name in HAR_DATASETS:
    print(f"\n{'='*60}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # Dictionary to accumulate results across all folds
    # Structure: {algo_name: [[fold1_scores], [fold2_scores], ...]}
    all_folds_data = {}

    try:
        # Loop through each fold to collect data
        for fold_num in range(1, NUM_FOLDS + 1):
            print(f"Loading fold {fold_num} for {dataset_name}...")
            directory = os.path.join(os.path.abspath("."), dataset_name, str(fold_num))

            try:
                results = algo_results_manager(dataset=dataset_name, target_subfolder=str(fold_num),
                                               file_name='algo_manager')

                for algo in results.algorithms:
                    if algo == "detached":
                        algo_predictions = results.predictions_dict[algo]
                    else:
                        algo_predictions = results.predictions_dict[algo]

                    # Extract scores for this fold
                    fold_scores = [algo_predictions[index] for index in range(len(features_array))]

                    # Initialize list for this algorithm if first time seeing it
                    if algo not in all_folds_data:
                        all_folds_data[algo] = []

                    all_folds_data[algo].append(fold_scores)

            except Exception as e:
                print(f"Warning: Could not load fold {fold_num} for {dataset_name}: {e}")
                continue

        # Calculate averages across folds for each algorithm
        averaged_data = {}
        for algo, folds_list in all_folds_data.items():
            # Convert to numpy array for easy averaging: shape (num_folds, num_feature_points)
            folds_array = np.array(folds_list)
            # Average across folds (axis=0)
            averaged_data[algo] = np.mean(folds_array, axis=0)
            print(f"  {algo}: averaged across {len(folds_list)} folds")

        # Create the plot
        plt.figure(figsize=(10, 6))

        # Plot each algorithm's performance with distinct colors
        # JMK-FS first (leftmost in legend)
        plt.plot(features_array, averaged_data["kmeans_avg_jm"], marker='d', color='#000000', label='JMK-FS', linewidth=2)  # Black
        plt.plot(features_array, averaged_data["fisher"], marker='o', color='#1f77b4', label='Fisher')         # Blue
        plt.plot(features_array, averaged_data["mrmr"], marker='s', color='#ff7f0e', label='MrMr')            # Orange
        plt.plot(features_array, averaged_data["relieff"], marker='^', color='#2ca02c', label='Relieff')      # Green
        plt.plot(features_array, averaged_data['random'], marker='*', color='#9467bd', label='Random')        # Purple
        plt.plot(features_array, averaged_data["detached"], marker='x', color='#17becf', label='Detached')    # Cyan

        # Customize the plot
        plt.xlabel('Number of Features Selected', fontsize=14)
        plt.ylabel('Classification Score', fontsize=14)
        plt.grid(True)
        plt.legend(fontsize=12, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=6)  # Legend below plot in single line
        plt.xticks(features_array, fontsize=14)  # Ensure all feature numbers are shown on x-axis
        plt.yticks(fontsize=14)

        # Display the plot
        plt.tight_layout()
        output_path = os.path.join(output_base_dir, f'{dataset_name}_averaged.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save with high quality
        plt.close()  # Close the figure to free memory
        print(f"Saved averaged plot: {output_path}\n")

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}\n")

print(f"\n{'='*60}")
print(f"All averaged plots saved to: {output_base_dir}")
print(f"{'='*60}")
