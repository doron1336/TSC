# 1/ pick a dataset
# 2/ use minirocket to create the transform
# 3/ calculate the JM - take the flattened version
# 4/ same process with shorter signal
# 5/ calculate gmm
import os
import pickle

import numpy as np

from MMD.distance_md import mmd, trim_and_rocket
from utils.JM import JM_flat
from utils.file_system import save_to_pickle
from utils.kmeans import perform_kmeans_clustering
from utils.retrieve_minirocket import retrieve_minirocket_data, load_ucr_dataset
from utils.topK_indices import calc_score

baseDir = "UCRArchive_2018"
har_dataset = 'HAR_datasets'
without_GA = 'without_GA'

os.chdir(os.path.join(os.path.dirname(os.getcwd()), baseDir))
HAR_DATASETS = ['GestureMidAirD3', 'GestureMidAirD2', 'UWaveGestureLibraryAll', 'GesturePebbleZ2', 'AllGestureWiimoteX',
                'CricketX', 'CricketY']
DIRECTORY_NUM: int = 3
for dataset_name in HAR_DATASETS:
    without_har = os.path.join(os.path.abspath("."), dataset_name)
    base_dir = os.path.join(os.path.abspath("."), har_dataset, dataset_name)
    directory = os.path.join(base_dir, str(DIRECTORY_NUM))
    minirocket_directory = os.path.join(os.path.abspath("."), dataset_name, str(DIRECTORY_NUM))
    print(directory)
    mmd_dir = os.path.join(directory, 'MMD')
    os.makedirs(directory, exist_ok=True)

    X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory=minirocket_directory,
                                                                                    dataset_name=dataset_name)
    prediction_score_full_signal = calc_score(X_train_transform, X_test_transform, y_train, y_test)
    print(prediction_score_full_signal)
    if os.path.exists(os.path.join(directory, 'JM_flat_data_full')):
        with open(os.path.join(directory, 'JM_flat_data_full'), 'rb') as file:
            JM_flat_data = pickle.load(file)
    else:
        JM_flat_data, _ = JM_flat(X_train_transform, y_train)
        save_to_pickle(data=JM_flat_data, directory=mmd_dir, file_name='JM_flat_data_full')
    x_train, y_train, x_test, y_test = load_ucr_dataset(without_har)

    num_features = x_train.shape[1]
    percentages = np.arange(10, 101, 10)  # 10%, ..., 100%
    x_train_range = [int(p * num_features / 100) for p in percentages]  # Convert percentages to feature counts

    mmd_result = []
    prediction_score_trimmed = []
    prediction_score_selected = []
    for i in x_train_range:
        print(i)
        X_train_transform_trimmed = trim_and_rocket(x_train, i)
        X_test_transform_trimmed = trim_and_rocket(x_test, i)
        prediction_score_trimmed.append(
            calc_score(X_train_transform_trimmed, X_test_transform_trimmed, y_train, y_test))
        if os.path.exists(os.path.join(mmd_dir, f"JM_trimmed_{i}")):
            with open(os.path.join(mmd_dir, f"JM_trimmed_{i}"), 'rb') as file:
                JM_flat_data_trimmed = pickle.load(file)
        else:
            JM_flat_data_trimmed, _ = JM_flat(X_train_transform_trimmed, y_train)
            save_to_pickle(data=JM_flat_data_trimmed, directory=mmd_dir, file_name=f"JM_trimmed_{i}")
        selected_features = perform_kmeans_clustering(JM_flat_data_trimmed, n_clusters=200)
        prediction_score_selected.append(calc_score(
            X_train_transform_trimmed[:, selected_features],
            X_test_transform_trimmed[:, selected_features],
            y_train,
            y_test))
        mmd_result.append(mmd(JM_flat_data, JM_flat_data_trimmed))
    with open(os.path.join(mmd_dir, 'arrays.pkl'), "wb") as f:
        pickle.dump({"prediction_score_trimmed": prediction_score_trimmed, "prediction_score_selected": prediction_score_selected,
                     "mmd_result": mmd_result}, f)

# Modified plotting code
# import matplotlib.pyplot as plt
#
# # Create plot with two y-axes and solid lines
# fig, ax1 = plt.subplots()
#
# # Plot MMD score on the left y-axis with solid line
# ax1.plot(percentages, mmd_result, '-o', label='MMD Score', color='red', markersize=5)
# ax1.set_xlabel('Percentage of Features Kept (%)')
# ax1.set_ylabel('MMD Score', color='red')
# ax1.tick_params(axis='y', labelcolor='red')
#
# # Create second y-axis for prediction scores
# ax2 = ax1.twinx()
# ax2.plot(percentages, prediction_score_trimmed, '-^', label='Prediction Score (Trimmed)', color='blue', markersize=5)
# ax2.plot(percentages, prediction_score_selected, '-s', label='Prediction Score (Selected)', color='green', markersize=5)
# ax2.set_ylabel('Prediction Score', color='blue')
# ax2.tick_params(axis='y', labelcolor='blue')
#
# # Add title
# plt.title('MMD and Prediction Scores vs Percentage of Features Kept')
#
# # Combine legends from both axes
# lines1, labels1 = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
#
# # Show plot
# plt.show()
