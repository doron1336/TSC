import pickle
import sys
import scipy.io as sio
import os
from os.path import dirname, join as pjoin
import numpy as np
import h5py
import pandas as pd
from sktime.datasets import load_osuleaf
from sktime.utils.validation.panel import check_X
from handmovementsDATA import GetHandMovementDATA
from models.GA import generations
from models.minirocket import fit, transform
from sklearn.linear_model import RidgeClassifierCV
from mrmr import mrmr_classif


# Load the Data
os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database")

with open("HandMovementDATA_xtrain", 'rb') as file:
    x_train = pickle.load(file)
with open("HandMovementDATA_ytrain", 'rb') as file:
    y_train = pickle.load(file)
with open("HandMovementDATA_xtest", 'rb') as file:
    x_test = pickle.load(file)
with open("HandMovementDATA_ytest", 'rb') as file:
    y_test = pickle.load(file)

# Rocket stuff
evaluated = False
filename_train = 'X_train_transform'
filename_test = 'X_test_transform'
# parameters = fit(x_train)
with open(filename_train, 'rb') as file:
    X_train_transform = pickle.load(file)
with open(filename_test, 'rb') as file:
    X_test_transform = pickle.load(file)

# X_train_transform = transform(x_train, parameters)
# X_test_transform = transform(x_test, parameters)
# if (not evaluated):
#     # print(X_train_transform)
#     classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
#     classifier.fit(X_train_transform, y_train)
#     pickle.dump(classifier, open(filename, 'wb'))

#    classifier = pickle.load(open(filename, "rb"))

# Classification
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

classifier.fit(X_train_transform, y_train)
predictions = classifier.score(X_test_transform, y_test)

# GA
chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800, n_feat=X_train_transform.shape[1], n_parents=640, mutation_rate=0.20, n_gen=2,
                                     X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train, Y_test=y_test)
print(len(chromo_df_bc))
total = 0
a = []
for chromo in chromo_df_bc:
    for i in chromo:
        total = total + i
    a.append(total)
    print("number of features in chromo", total)
    total = 0

print("GA score", score_bc)
print("number of features", total)

print("prediction score without GA", predictions)

# Generate an array with x True values in random locations
array_size = 9996
for i in a:
    random_indices = np.random.choice(array_size, i, replace=False)
    selected_elements = np.zeros(array_size, dtype=bool)
    selected_elements[random_indices] = True

    classifier_selected_randomly = RidgeClassifierCV(
        alphas=np.logspace(-3, 3, 10))
    classifier_selected_randomly.fit(
        X_train_transform[:, selected_elements], y_train)
    predictions = classifier_selected_randomly.score(
        X_test_transform[:, selected_elements], y_test)
    print(
        f"Rocket score with randomly {i} selected features", predictions)

# mrmr over GA
# select top 10 features using mRMR
# we will use the features after first generation GA
X = X_train_transform[:, chromo_df_bc[0]]
selected_features_mrmr = mrmr_classif(X=X, y=y_train, K=100)
print(selected_features_mrmr)
# classifier_mrmr = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# classifier_mrmr.fit(X_train_transform[:, selected_features_mrmr], y_train)
# predictions = classifier_selected_randomly.score(X_test_transform[:, selected_features_mrmr], y_test)
