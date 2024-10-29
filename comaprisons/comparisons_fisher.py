# from skfeature.function.similarity_based import fisher_score
import numpy as np
from scipy.sparse import *
from skfeature.utility.construct_W import construct_W
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

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


def fisher_score(X, y):
    """
    This function implements the fisher score feature selection, steps are as follows:
    1. Construct the affinity matrix W in fisher score way
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1

    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels

    Output
    ------
    score: {numpy array}, shape (n_features,)
        fisher score for each feature

    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.
    """

    # Construct weight matrix W in a fisherScore way
    kwargs = {"neighbor_mode": "supervised", "fisher_score": {True}, 'y': y}
    W = construct_W(X, **kwargs)

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp) / D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp) / D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1 / D_prime))[0, :]

    # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
    score = 1.0 / lap_score - 1
    return np.transpose(score)


def feature_ranking(score, k):
    """
    Rank features in descending order according to fisher score, the larger the fisher score, the more important the
    feature is
    """
    sorted_indices = np.argsort(score, 0)
    # Get the indices of the top 50 elements (last 50 indices)
    top_k_indices = sorted_indices[-k:]
    return top_k_indices


@record_duration
def fisher_ranking(train, target, num_of_features):
    selector = SelectKBest(score_func=f_classif, k=num_of_features)
    X_new = selector.fit_transform(train, target)
    score = selector.scores_
    return feature_ranking(score, num_of_features)

# returns rank directly instead of fisher score. so no need for feature_ranking

# n_features_to_keep = 50
# selector = SelectKBest(score_func=f_classif, k=n_features_to_keep)
# X_new = selector.fit_transform(Rocket_output_train, y_train)
# score = selector.scores_
# # print(score)
# # score = fisher_score(Rocket_output_train, y_train)

# selected_features = feature_ranking(score, n_features_to_keep)

# classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# classifier_selected.fit(Rocket_output_train[:, selected_features], y_train)
# predictions = classifier_selected.score(
#     Rocket_output_test[:, selected_features], y_test)
# print(
#     f"fisher score with {n_features_to_keep} selected features", predictions)
