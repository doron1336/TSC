from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os
import pickle as pkl
# import matplotlib.pyplot as plt
import pandas as pd
from mrmr import mrmr_classif
from utils.timit import record_duration


# # using mrmr as a filter method
# os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database")
#
# with open("X_test_transform", "rb") as f:
#     Rocket_output_test = pkl.load(f)
# with open("HandMovementDATA_ytest", "rb") as f:
#     y_test = pkl.load(f)
#
# with open("X_train_transform", "rb") as f:
#     Rocket_output_train = pkl.load(f)
# with open("HandMovementDATA_ytrain", "rb") as f:
#     y_train = pkl.load(f)

@record_duration
def mrmr_ranking(train, target, num_of_features):
    return mrmr_classif(X=pd.DataFrame(train), y=pd.Series(target), K=num_of_features)


# n_features_to_keep = 50
# selected_features = mrmr_classif(X=pd.DataFrame(
#     Rocket_output_train), y=pd.Series(y_train), K=n_features_to_keep)
# print(selected_features)

# classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# classifier_selected.fit(Rocket_output_train[:, selected_features], y_train)
# predictions = classifier_selected.score(
#     Rocket_output_test[:, selected_features], y_test)
# print(
#     f"mRMR score with {n_features_to_keep} selected features", predictions)
