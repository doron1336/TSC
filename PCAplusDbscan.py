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
import pickle

# load DATA
x_train, y_train, x_test, y_test = GetHandMovementDATA()

# miniRocket section
filename_train = os.path.abspath(".") + "\\X_train_transform.pkl"
filename_test = os.path.abspath(".") + "\\X_test_transform.pkl"

parameters = fit(x_train)
X_train_transform = transform(x_train, parameters)
X_test_transform = transform(x_test, parameters)
print(X_train_transform.shape)
# data, dataMean = JM_flat(X_train_transform, y_train)
# print("size of data from JM_flat", data.shape)

# with open(filename_train, 'wb') as file:
#     pickle.dump(X_train_transform, file)
# with open(filename_test, 'wb') as file:
#     pickle.dump(X_test_transform, file)

# with open(filename_train, 'rb') as file:
#     X_train_transform = pickle.load(file)
# with open(filename_test, 'rb') as file:
#     X_test_transform = pickle.load(file)

# with open(filename_train, 'wb') as f:
#     np.save(f, X_train_transform)
# with open(filename_train, 'wb') as f:
#     np.save(f, X_test_transform)

# with open(filename_train, 'rb') as f:
#     X_train_transform = np.load(f)
# with open(filename_train, 'rb') as f:
#     X_test_transform = np.load(f)

print(X_train_transform.shape)

# GA

filename_chromo = os.path.abspath(".") + "\\selectedChromo.pkl"
chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800, n_feat=X_train_transform.shape[1], n_parents=640, mutation_rate=0.20, n_gen=2,
                                     X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train, Y_test=y_test)
print(len(chromo_df_bc))
max = 0
selectedChromo = np.empty(X_train_transform.shape[1])
for chromo in chromo_df_bc:
    numOfSelectedFeatures = np.sum(chromo)
    if numOfSelectedFeatures > max:
        max = numOfSelectedFeatures
        selectedChromo = chromo
    print("number of features in chromo", numOfSelectedFeatures)

print("GA score", score_bc)
with open(filename_chromo, 'wb') as file:
    pickle.dump(selectedChromo, file)

# with open(filename_chromo, 'rb') as f:
#     selectedChromo = np.load(f)

new_X_train_transform = X_train_transform[:, selectedChromo]
new_X_test_transform = X_test_transform[:, selectedChromo]

print(new_X_train_transform.shape)
data, dataMean = JM_flat(new_X_train_transform, y_train)
print(data.shape)
array_size = new_X_train_transform.shape[1]
# for i in range(1, 15):
#     num_true_values = int(0.02*i*array_size)
#     random_indices = np.random.choice(
#         array_size, num_true_values, replace=False)
#     selected_elements = np.zeros(array_size, dtype=bool)
#     selected_elements[random_indices] = True

#     classifier_selected_randomly = RidgeClassifierCV(
#         alphas=np.logspace(-3, 3, 10))
#     classifier_selected_randomly.fit(
#         new_X_train_transform[:, selected_elements], y_train)
#     predictions = classifier_selected_randomly.score(
#         new_X_test_transform[:, selected_elements], y_test)
#     print(
#         f"With randomly {num_true_values} selected features", predictions)


# PCA plus DBSCAN
pca = PCA(n_components=2).fit(data)
pca_result = pca.transform(data)
print(type(pca_result))
# filename = os.path.abspath(".") + "\\pca_result.npy"
# np.save(filename, pca_result)

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
    numOfElements = math.ceil(0.02*count) if k == -1 else math.ceil(0.01*count)
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

# classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

# classifier.fit(X_train_transform, y_train)
# predictions = classifier.score(X_test_transform, y_test)
# print("Rocket score", predictions)

# # ROCKET WITH FEATURES SELECTED BY DBSCAN
classifier_selected_dbscan = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

classifier_selected_dbscan.fit(
    X_train_transform[:, all_feature_locations], y_train)
predictions = classifier_selected_dbscan.score(
    X_test_transform[:, all_feature_locations], y_test)
print(
    f"Rocket score with {len(all_feature_locations)} dbscan selected features", predictions)
