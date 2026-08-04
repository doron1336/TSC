import math
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from models.minirocket import fit, transform
from utils.JM import JM_flat
from handmovementsDATA import GetHandMovementDATA
from sklearn.decomposition import PCA
from models.GA import generations
from sklearn.linear_model import RidgeClassifierCV
from sklearn import metrics
from sklearn.cluster import DBSCAN
import seaborn as sns
import pickle


x_train, y_train, x_test, y_test = GetHandMovementDATA()

parameters = fit(x_train)
X_train_transform = transform(x_train, parameters)
data, dataMean = JM_flat(X_train_transform, y_train)
filename = os.path.abspath(".") + "\\X_train_transform.npy"
np.save(filename, X_train_transform)

pca = PCA(n_components=2).fit(data)
pca_result = pca.transform(data)
print(type(pca_result))
filename = os.path.abspath(".") + "\\pca_result.npy"
np.save(filename, pca_result)

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
    # print(class_member_mask)
    # print(core_samples_mask)
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

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

X_test_transform = transform(x_test, parameters)
classifier.fit(X_train_transform, y_train)
predictions = classifier.score(X_test_transform, y_test)
print("Rocket score", predictions)

# ROCKET WITH FEATURES SELECTED BY DBSCAN
classifier_selected_dbscan = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

classifier_selected_dbscan.fit(
    X_train_transform[:, all_feature_locations], y_train)
predictions = classifier_selected_dbscan.score(
    X_test_transform[:, all_feature_locations], y_test)
print(
    f"Rocket score with {len(all_feature_locations)} dbscan selected features", predictions)


# ROCKET WITH FEATURES SELECTED randomly
# Size of the array
array_size = 9996

# Number of True values to generate
num_true_values = len(all_feature_locations)

# Generate an array with x True values in random locations
random_indices = np.random.choice(array_size, num_true_values, replace=False)
selected_elements = np.zeros(array_size, dtype=bool)
selected_elements[random_indices] = True

classifier_selected_randomly = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier_selected_randomly.fit(
    X_train_transform[:, selected_elements], y_train)
predictions = classifier_selected_randomly.score(
    X_test_transform[:, selected_elements], y_test)
print(
    f"Rocket score with randomly {num_true_values} selected features", predictions)


# GA WITH FEATURES SELECTED BY DBSCAN
X_train_selected = X_train_transform[:, all_feature_locations]
print(f"using GA with {len(all_feature_locations)} features")
chromo_df_bc, score_bc = generations(X_train_selected, y_train, size=800, n_feat=X_train_selected.shape[1], n_parents=640, mutation_rate=0.20, n_gen=4,
                                     X_train=X_train_selected, X_test=X_test_transform[:, all_feature_locations], Y_train=y_train, Y_test=y_test)
print(len(chromo_df_bc))
total = 0
for chromo in chromo_df_bc:
    for i in chromo:
        total = total + i
    print("number of features in chromo", total)
    total = 0

print("GA score", score_bc)
