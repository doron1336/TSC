import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import os
os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\bodyMovement")
print(os.listdir())
train = pd.read_csv("train.csv")
# train.head()

X = train.iloc[:, :-2]
# X.head()

y_train = train.iloc[:, -1]
print(f" Classes: {set(y_train)} ")
y_train.value_counts()


modelo = RandomForestClassifier(n_estimators=100)
modelo_GB = GradientBoostingClassifier()

modelo.fit(X, y_train)

test = pd.read_csv("test.csv")
X_test = test.iloc[:, :-2]
y_test = test.iloc[:, -1]

# sensor = np.expand_dims(X_test.iloc[0, :], axis=1)
# predecir_sensor = modelo.predict(sensor.T)

pred_gb = modelo.predict(X_test)
print(classification_report(y_test, pred_gb))
