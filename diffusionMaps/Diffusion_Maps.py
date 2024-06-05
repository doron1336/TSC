# coding: utf-8
'''
The program calculates diffusion maps coordinates for vectors presented as rows in a file defined by a variable named <data>.
Resulting diffusion maps coordinates are stored in a variable named <coordinates>.
The number of diffusion coordinates computed is hard coded in a variable named <dim> (currently dim=5).
The top diffusion maps coordinates are typically used to embed the original data into a low dimensional Euclidean space.

 The basic steps are:
 Given N m-dimensional data points (here these are flattened sonograms) do
 1) Compute an N by N matrix that holds the pairwise distances as defined by a Gaussian kernel
 2) Normalize the kernel to be row-stochastic (sum of each row = 1)
 3) Compute the eigenvectors and eigenvalues of the normalized kernel
 4) Use the top left eigenvectors to enbed the data points. Here we use the 2st and 4rd for plotting.
 Each eigenvector is of size NX1, the ith entry corresponds to input point number i.


Last modified by Neta Rabin on 2019/02/27.
'''
# Add the parent directory of mypackage to the Python path
from sklearn.linear_model import RidgeClassifierCV
from sklearn.cluster import KMeans
import pickle as pkl
import sys
import os
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from numpy import linalg as LA
import numpy as np
from utils.timit import record_duration

import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector

'''
epsilon_factor - a parameter that controls the width of the Gaussian kernel  
'''
epsilon_factor = 4

'''
Compute  the width of the Gaussian kernel based on the given dataset.   
'''

# compute epsilon of (dataList)


def calcEpsilon(dataList, eps_type, epsilon_factor=5):

    dist = squareform(pdist(dataList))  # read about squareform
    if eps_type == 'maxmin':
        # option #1 - epsilon= max min(distance)
        idist = dist+np.identity(len(dist))
        eps_maxmin = np.max(np.min(idist, axis=0))
        epsilon = eps_maxmin*epsilon_factor
    elif eps_type == 'mean':
        # option #2 - epsilon= mean(distance)
        eps_mean = np.mean(dist)
        epsilon = eps_mean * epsilon_factor

    else:
        raise KeyError('eps_type should be either maxmin or mean')
    return dist, epsilon


def ker_calc(dataList, eps_type):
    dist, eps = calcEpsilon(dataList, eps_type, epsilon_factor=1)
    ker = np.exp(-(dist**2) / (2*eps))
    return ker, eps


'''
Construct the NXN Gaussian kernel, normalize it and compute eigenvalues and eigenvectors.   
'''


def diffusionMapping(dataList, alpha, eps_type, t, **kwargs):
    try:
        kwargs['dim'] or kwargs['delta']
    except KeyError:
        raise KeyError('specify either dim or delta as keyword argument!')

    # compute epsilon of (dataList) eps_type can be 'mean' or 'maxmin'
    # dist is the L2 distances
    # eps_type='maxmin'#mean' #or maxmin

    ker, epsilon = ker_calc(dataList, eps_type)
    v = np.sum(ker, axis=0)

    v = v**alpha
    V_x_y = v*v[:, None]
    a = ker/V_x_y


# calc the row sums of a, save as v1
# in the next for-loop, divide the rows of a by v1
    sa = np.sum(a, axis=0)
    m = a/sa[:, None]

    # compute eigenvectors of (a_ij)
    vecs, eigs, _ = LA.svd(m, full_matrices=False)
    # vecs = vecs / vecs[:, 0][:, None]

    # Compute dimension
    # (for better performance you may want to combine this with an iterative way of computing eigenvalues/vectors)
    if kwargs['dim']:
        embeddim = kwargs['dim']
    elif kwargs['delta']:
        i = 1
        while LA.eigvals[i] ** t > kwargs['delta'] * LA.eigvals[1] ** t:
            i += 1
        embeddim = i

    # Compute embedding coordinates
    diffusion_coordinates = vecs[:, 1:embeddim +
                                 1].T * (eigs[1:embeddim + 1][:, None] ** t)
    # print(f"epsilon={epsilon}")

    return (vecs, eigs, diffusion_coordinates.T, dataList, epsilon)


@record_duration
def dm_ranking(data, num_of_features, q):

    avg_jm = np.mean(data, axis=1)
    eps_type = 'mean'  # mean' #or maxmin
    alpha = 1
    vecs, eigs, coordinates, dataList, epsilon = diffusionMapping(
        data, alpha, eps_type, 1, dim=3)  # dim - number of diffusion coordinates computed

    # Pick best features
    sorted_indices = np.argsort(-avg_jm)
    # print(sorted_indices)
    # Calculate the index corresponding to the q percentile
    index_q_percentile = int(len(data) * q / 100)
    top_q_percent_indices = sorted_indices[:index_q_percentile]

    # K-Means
    kmeans = KMeans(init="random", n_clusters=num_of_features,
                    max_iter=300, n_init=5)
    label = kmeans.fit_predict(coordinates[top_q_percent_indices])
    u_labels = np.unique(label)
    # Pick best features
    selected_features = []
    for i in u_labels:
        arr = np.copy(avg_jm)
        indices_to_exclude = np.where(label == i)
        value_to_set = -10
        mask = np.ones_like(arr, dtype=bool)
        mask[indices_to_exclude] = False
        arr[mask] = value_to_set
        selected_features.append(np.argmax(arr))
    return selected_features, coordinates


@record_duration
def dm_ranking_datafold(data, num_of_features, q):

    avg_jm = np.mean(data, axis=1)
   
    # Optimize kernel parameters
    X_pcm = pfold.PCManifold(data)
    X_pcm.optimize_parameters(result_scaling=3)

    print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

    # Diffusion Maps
    dmap = dfold.DiffusionMaps(
        kernel=pfold.GaussianKernel(
            epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
        ),
        n_eigenpairs=9,
    )
    dmap = dmap.fit(X_pcm)
    evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_
    coordinates = evecs[:,[1,2]]
    # Pick best features
    sorted_indices = np.argsort(-avg_jm)
    # print(sorted_indices)
    # Calculate the index corresponding to the q percentile
    index_q_percentile = int(len(data) * q / 100)
    top_q_percent_indices = sorted_indices[:index_q_percentile]

    # K-Means
    kmeans = KMeans(init="random", n_clusters=num_of_features,
                    max_iter=300, n_init=5)
    label = kmeans.fit_predict(coordinates[top_q_percent_indices])
    u_labels = np.unique(label)
    # Pick best features
    selected_features = []
    for i in u_labels:
        arr = np.copy(avg_jm)
        indices_to_exclude = np.where(label == i)
        value_to_set = -10
        mask = np.ones_like(arr, dtype=bool)
        mask[indices_to_exclude] = False
        arr[mask] = value_to_set
        selected_features.append(np.argmax(arr))
    return selected_features, coordinates


# Classification

# classifier_selected_dm = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# with open("X_train_transform", "rb") as f:
#     X_train_transform = pkl.load(f)
# with open("X_test_transform", "rb") as f:
#     X_test_transform = pkl.load(f)
# with open("HandMovementDATA_ytrain", "rb") as f:
#     y_train = pkl.load(f)
# with open("HandMovementDATA_ytest", "rb") as f:
#     y_test = pkl.load(f)

# classifier_selected_dm.fit(X_train_transform[:, features], y_train)
# predictions = classifier_selected_dm.score(
#     X_test_transform[:, features], y_test)
# print(
#     f"Rocket score with {len(features)} dbscan selected features", predictions)
