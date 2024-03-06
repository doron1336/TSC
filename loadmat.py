import os
from os.path import dirname, join as pjoin
import numpy as np
import pandas as pd
import scipy.io as sio

from models import minirocket
from utils.JM import JM_flat
from handmovementDATA_randomTest import GetHandMovementDATA_randomTest
from models.GA import generations
from sklearn.linear_model import RidgeClassifierCV


# os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\ninapro_dataset")
# mat = sio.loadmat('S1_E1_A1.mat')
# mat_to_pd = pd.Series(mat)
# print(mat_to_pd.head)
# sub_db_emg = mat_to_pd["emg"] 
# print(sub_db_emg.shape)
# lables_maybe = mat_to_pd["stimulus"]
# print(lables_maybe.shape)
# print(lables_maybe[1797051])

os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database\yuri")
mat = sio.loadmat('HRFI.BHZ.200701021000.mat')
mat2 = sio.loadmat('HRFI.BHZ.200809020856.mat')
mat_to_pd = pd.Series(mat)
mat_to_pd2 = pd.Series(mat2)
# print(mat_to_pd.head)
# print(len(mat["W"]))
x = np.hstack(mat["W"])
print(x)
# x_train = np.vstack([x[:3000], np.hstack(mat2["W"])[:3000]])
# print(len(x_train[:3000]))
# print(len(mat_to_pd["w"]))
# print(len(mat_to_pd.values))

# applying minirocket
# parameters = minirocket.fit(x_train)
# X_train_transform = minirocket.transform(x_train, parameters)
# X_test_transform = minirocket.transform(x_test, parameters)
# print(X_train_transform.shape)