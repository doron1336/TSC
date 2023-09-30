import os
from comparisons_fisher import fisher_ranking
from comparisons_mrmr import mrmr_ranking
from comparisons_relieff import relieff_ranking
from diffusionMaps.Diffusion_Maps import dm_ranking
import pickle
from sklearn.linear_model import RidgeClassifierCV
import numpy as np

# Load the Data
os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database\osuleaf")

# miniRocket section
filename_train = os.path.abspath(".") + "\\osuleaf_train"
filename_test = os.path.abspath(".") + "\\osuleaf_test"

with open("osuleaf_minirocket_train", 'rb') as file:
    X_train_transform = pickle.load(file)
with open("osuleaf_minirocket_test", 'rb') as file:
    X_test_transform = pickle.load(file)
with open("osuleaf_y_train", 'rb') as file:
    y_train = pickle.load(file)
with open("osuleaf_y_test", 'rb') as file:
    y_test = pickle.load(file)
with open("JM_FLAT_osuleaf", 'rb') as file:
    JM_FLAT_osuleaf = pickle.load(file)


def random_choosing(num_of_features):
    array_size = 9996
    random_indices = np.random.choice(
        array_size, num_of_features, replace=False)
    # selected_elements = np.zeros(array_size, dtype=bool)
    # selected_elements[random_indices] = True
    return random_indices


array_dict = {key: [] for key in [0, 1, 2, 3, 4]}
dict = {0: "fisher", 1: "mrmr", 2: "relief", 3: "dm", 4: "random"}
num_features = []
for i in range(10, 210, 10):
    num_features.append(i)
    fishers_selected = fisher_ranking(X_train_transform, y_train, i)
    mrmr_selected = mrmr_ranking(X_train_transform, y_train, i)
    relief_selected = relieff_ranking(X_train_transform, y_train, i)
    dm_selected = dm_ranking(JM_FLAT_osuleaf, i)
    random_selected = random_choosing(i)
    for j in range(5):
        a = {0: fishers_selected, 1: mrmr_selected,
             2: relief_selected, 3: dm_selected, 4: random_selected}
        classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier_selected.fit(
            X_train_transform[:, a[j]], y_train)
        predictions = classifier_selected.score(
            X_test_transform[:, a[j]], y_test)
        array_dict[j].append(predictions)

for m in range(5):
    pickle.dump(array_dict[m], open(dict[m], 'wb'))
