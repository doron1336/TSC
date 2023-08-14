import math
import sys
from matplotlib import pyplot as plt
import numpy as np
from torch import optim
from models.minirocket import fit, transform
from utils.JM import JM_flat
from handmovementsDATA import GetHandMovementDATA
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.cluster import DBSCAN
import seaborn as sns
from models.GA import *

pca_result = np.load("handMovement\Database\pca_result.npy")
all_feature_locations = np.empty(0)

db = DBSCAN(eps=0.13, min_samples=10).fit(pca_result)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

unique_labels = set(labels)

core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
print(len(db.core_sample_indices_))

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k
    print(class_member_mask)
    print(core_samples_mask)
    count = np.count_nonzero(class_member_mask)
    feature_locations = np.asarray(class_member_mask == True).nonzero()[0]
    numOfElements = math.ceil(0.3*count) if k == -1 else math.ceil(0.2*count)
    selected_elements = np.random.choice(
        feature_locations, size=numOfElements, replace=False)
    all_feature_locations = np.append(
        all_feature_locations, selected_elements).astype(np.int64)
    xy = pca_result[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    xy = pca_result[class_member_mask & ~core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

print(len(all_feature_locations))
plt.title(f"Estimated number of clusters: {n_clusters_}")
plt.show()
