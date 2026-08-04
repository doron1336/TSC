# from tslearn.datasets import UCR_UEA_datasets
# from sktime.transformations.panel.rocket import MiniRocket
from enum import unique
from re import T
from tkinter import Y
import numpy as np
from sktime.utils.validation.panel import check_X
from sktime.transformations.panel.rocket import Rocket
from minirocket import MiniRocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.datasets import load_arrow_head
import pandas as pd
import matplotlib.pyplot as plt
import pylab
import obspy
from obspy.signal.trigger import plot_trigger, recursive_sta_lta
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


path = "./neta_data_unfiltered"

data3 = pd.read_csv("data3.csv")
event_id = data3["event_id"] 
unique_event_ids = set(event_id)

train_events = list(unique_event_ids)[0:24]
test_events = list(unique_event_ids)[25:]


try_event = train_events[0]
# print(unique_event_ids)

records = data3[data3["event_id"] == try_event]["record_name"].tolist()

# print(record)
# X = np.zeros(shape=(5, 400))
# x = []
# for i in range(5):
#     st = obspy.read(f"{path}/{records[i]}")
#     tr = st.select(component="Z")[0]
#     df = tr.data[800:1200].tolist()
#     x.append(df)
# a = np.asarray(x)
# print(type(a))
# print(a)
# print(a.shape)
a = []
y_train = []
for i in train_events:
    event = i
    record = data3[data3["event_id"] == i]["record_name"].tolist()
    for j in record:
        if j == "202206100139_EIL.mseed":
            continue              
        st = obspy.read(f"{path}/{j}")  # load example seismogram
        dist = data3[data3["record_name"] == j]["dist"].tolist()[0]
        tr = st.select(component="Z")[0]
        df = tr.data[800:1200]
        if len(df) != 400:
            print(j)
        a.append(df)
        y_train.append(dist)

X = np.vstack(a)
Y = np.asanyarray(y_train)
print(Y)

minirocket = MiniRocket()

parameters = minirocket.fit(X.astype(np.float32))

print(parameters)
X_train_transform = minirocket.transform(X.astype(np.float32), parameters)
print(X_train_transform.shape)

np.save('seismoMinirocket.npy', X_train_transform)


# Clustering using TSNE
# model = TSNE(learning_rate=100)
# transformed = model.fit_transform(X_train_transform) 
# # Plotting 2d t-Sne
# x_axis = transformed[:, 0]
# y_axis = transformed[:, 1]

# plt.scatter(x_axis, y_axis)
# plt.show()

# Transoring Using PCA

# Declaring Model
dbscan = DBSCAN()
dbscan.fit(X_train_transform)
pca = PCA(n_components=2).fit(X_train_transform)
pca_2d = pca.transform(X_train_transform)
print(pca_2d.shape)

# # Plot based on Class
# for i in range(0, pca_2d.shape[0]):
#     if dbscan.labels_[i] == 0:
#         c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
#     elif dbscan.labels_[i] == 1:
#         c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
#     elif dbscan.labels_[i] == -1:
#         c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

# plt.legend([c1, c2, c3], ['Cluster 1', 'Cluster 2', 'Noise'])
# plt.title('DBSCAN finds 2 clusters and Noise')
# plt.show()


# classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
# classifier.fit(X_train_transform, y_train)

# X_test_transform = minirocket.transform(x_test, parameters)
# predictions = classifier.score(X_test_transform, y_test)
# print(predictions)