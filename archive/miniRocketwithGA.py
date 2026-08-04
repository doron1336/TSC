# from tslearn.datasets import UCR_UEA_datasets
from random import randint
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sktime.datasets import load_arrow_head, load_osuleaf

from models.GA import *
from models.minirocket import fit, transform
from utils.JM import *

x_train, y_train = load_arrow_head(split="test", return_X_y=True)
x_test, y_test = load_arrow_head(split="train", return_X_y=True)

# x_train, y_train = load_osuleaf(split="train", return_X_y=True)
# x_test, y_test = load_osuleaf(split="test", return_X_y=True)

# labels, counts = np.unique(y_train, return_counts=True)
# print(labels, counts)
parameters = fit(x_train)
X_train_transform = transform(x_train, parameters)
X_test_transform = transform(x_test, parameters)

print(type(X_train_transform))
print(X_train_transform.shape)

logmodel = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

# chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800, n_feat=X_train_transform.shape[1], n_parents=640, mutation_rate=0.20, n_gen=1,
#  X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train, Y_test=y_test)
# print(len(chromo_df_bc))
# total = 0
# for chromo in chromo_df_bc:
#     for i in chromo:
#         total = total + i
#     print("number of features in chromo", total)
#     total = 0
total = 1721


# Test against same number of features that was selected randomly
X_train = X_train_transform
X_test = X_test_transform
n_feat = X_train_transform.shape[1]
Y_train = y_train
Y_test = y_test
chromosome = np.ones(n_feat, dtype=np.bool)
chromosome[:int(n_feat-total)] = False
logmodel.fit(X_train[:, chromosome], Y_train)
predictions = logmodel.predict(X_test[:, chromosome])
score = accuracy_score(Y_test, predictions)
print(f"SCORE randomly selected {total} number of features - ", score)
# plot(score_bc, 0.9, 1.0, c="gold")
