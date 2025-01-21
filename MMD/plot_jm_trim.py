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

os.chdir(os.path.join(os.path.dirname(os.getcwd()), baseDir))
dataset_name: str = 'GestureMidAirD2'
DIRECTORY_NUM: int = 4
base_dir = os.path.join(os.path.abspath("."), dataset_name)
directory = os.path.join(base_dir, str(DIRECTORY_NUM))
print(directory)

X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)
prediction_score = calc_score(X_train_transform, X_test_transform, y_train, y_test)
print(prediction_score)
# JM_flat_data, _ = JM_flat(X_train_transform, y_train)
if os.path.exists(os.path.join(directory, without_GA, 'JM_flat_data_full')):
    with open(os.path.join(directory, without_GA, 'JM_flat_data_full'), 'rb') as file:
        JM_flat_data_full = pickle.load(file)
x_train, y_train, x_test, y_test = load_ucr_dataset(base_dir)

feature_to_keep = 150

X_train_transform_trimmed = trim_and_rocket(x_train, feature_to_keep)
X_test_transform_trimmed = trim_and_rocket(x_test, feature_to_keep)
prediction_score = calc_score(X_train_transform_trimmed, X_test_transform_trimmed, y_train, y_test)
print(prediction_score)
if os.path.exists(
        os.path.join(f'/Users/doron/Desktop/personal/thesis/TSC/MMD/{dataset_name}', f"JM_trimmed_{feature_to_keep}")):
    with open(os.path.join(f'/Users/doron/Desktop/personal/thesis/TSC/MMD/{dataset_name}',
                           f"JM_trimmed_{feature_to_keep}"), 'rb') as file:
        JM_flat_data_trimmed = pickle.load(file)
else:
    JM_flat_data_trimmed, _ = JM_flat(X_train_transform, y_train)
    save_to_pickle(JM_flat_data_trimmed, f'/Users/doron/Desktop/personal/thesis/TSC/MMD/{dataset_name}',
                   f"JM_trimmed_{feature_to_keep}")

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
plt.title(f't-SNE Visualization JM-full & JM-trimmed {feature_to_keep}')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.grid(True)
plt.show()
