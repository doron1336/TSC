import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from MMD.distance_md import trim_and_rocket
from utils.JM import JM_flat
from utils.file_system import save_to_pickle
from utils.retrieve_minirocket import retrieve_minirocket_data, load_ucr_dataset
from utils.topK_indices import calc_score

baseDir = "UCRArchive_2018"
without_GA = 'without_GA'
har_dataset = 'HAR_datasets'
DIRECTORY_NUM: int = 3

os.chdir(os.path.join(os.path.dirname(os.getcwd()), baseDir))
HAR_DATASETS = ['GestureMidAirD3', 'GestureMidAirD2', 'UWaveGestureLibraryAll', 'GesturePebbleZ2', 'AllGestureWiimoteX',
                'CricketX', 'CricketY']
for dataset_name in HAR_DATASETS:
    without_har = os.path.join(os.path.abspath("."), dataset_name)
    base_dir = os.path.join(os.path.abspath("."), har_dataset, dataset_name)
    directory = os.path.join(base_dir, str(DIRECTORY_NUM))
    mmd_dir = os.path.join(directory, 'MMD')
    print(directory)

    X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(os.path.join(without_har, str(DIRECTORY_NUM)), dataset_name)
    prediction_score = calc_score(X_train_transform, X_test_transform, y_train, y_test)
    print(prediction_score)
    x_train, y_train, x_test, y_test = load_ucr_dataset(without_har)

    percentage_to_keep = 30
    num_features = x_train.shape[1]
    feature_to_keep = int(percentage_to_keep * num_features / 100)

    if os.path.exists(os.path.join(mmd_dir, 'JM_flat_data_full')) and os.path.exists(os.path.join(mmd_dir, f"JM_trimmed_{feature_to_keep}")):
        with open(os.path.join(mmd_dir, 'JM_flat_data_full'), 'rb') as file:
            JM_flat_data_full = pickle.load(file)
        with open(os.path.join(mmd_dir, f"JM_trimmed_{feature_to_keep}"), 'rb') as file:
            JM_flat_data_trimmed = pickle.load(file)
    else:
        JM_flat_data_full, _ = JM_flat(X_train_transform, y_train)
        # save_to_pickle(data=JM_flat_data_full, directory=mmd_dir, file_name='JM_flat_data_full')
        X_train_transform_trimmed = trim_and_rocket(x_train, feature_to_keep)
        # X_test_transform_trimmed = trim_and_rocket(x_test, feature_to_keep)
        JM_flat_data_trimmed, _ = JM_flat(X_train_transform_trimmed, y_train)
        # save_to_pickle(data=JM_flat_data_trimmed, directory=mmd_dir, file_name=f"JM_trimmed_{feature_to_keep}")


    # prediction_score = calc_score(X_train_transform_trimmed, X_test_transform_trimmed, y_train, y_test)
    # print(prediction_score)

    ### Plot using TSNE Section

    # Concatenate the vectors
    data = np.vstack([JM_flat_data_full, JM_flat_data_trimmed])

    # Create labels (0 for JM_flat_data_full, 1 for JM_flat_data_trimmed)
    labels = np.array([0] * JM_flat_data_full.shape[0] + [1] * JM_flat_data_trimmed.shape[0])

    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    data_tsne = tsne.fit_transform(data_scaled)

    # Plot the results with color coding
    plt.figure(figsize=(10, 8))
    for label, color in zip([0, 1], ['blue', 'red']):
        if label == 0:
            label_name = "JM_flat_data_full"
        else:
            label_name = "JM_flat_data_trimmed"
        plt.scatter(
            data_tsne[labels == label, 0],
            data_tsne[labels == label, 1],
            c=color,
            label=label_name,
            s=10,
            alpha=0.7
        )
    plt.title(f't-SNE Visualization JM-full & JM-trimmed {percentage_to_keep}%')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    plt.grid(True)
    # Save the plot *before* showing or closing
    output_path = os.path.join(mmd_dir, f'{dataset_name}_plot_tsne_{percentage_to_keep}%.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Added dpi and bbox_inches for better quality
    plt.show()  # Display the plot (optional, can be removed in non-interactive scripts)
    plt.close()  # Close the figure to free memory
