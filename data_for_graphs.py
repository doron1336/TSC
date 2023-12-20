import os
from comparisons_fisher import fisher_ranking
from comparisons_mrmr import mrmr_ranking
from comparisons_relieff import relieff_ranking
from diffusionMaps.Diffusion_Maps import dm_ranking
import pickle
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from utils.JM import JM_flat
from models.GA import generations

# Load the Data
os.chdir(r"C:\Users\doron\OneDrive\Desktop\thesis\TSC\handMovement\Database\handmovement2\1")
dataset_name = "handmovement"
# miniRocket section
# filename_train = os.path.abspath(".") + "\\osuleaf_train"
# filename_test = os.path.abspath(".") + "\\osuleaf_test"
with open(f"{dataset_name}_minirocket_train", 'rb') as file:
    X_train_transform = pickle.load(file)
with open(f"{dataset_name}_minirocket_test", 'rb') as file:
    X_test_transform = pickle.load(file)
with open(f"{dataset_name}_y_train", 'rb') as file:
    y_train = pickle.load(file)
with open(f"{dataset_name}_y_test", 'rb') as file:
    y_test = pickle.load(file)
# with open("JM_FLAT_arrowhead", 'rb') as file:
    # JM_flat_data = pickle.load(file)


def random_choosing(num_of_features):
    array_size = 9996
    random_indices = np.random.choice(
        array_size, num_of_features, replace=False)
    # selected_elements = np.zeros(array_size, dtype=bool)
    # selected_elements[random_indices] = True
    return random_indices


# GA section
chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800, n_feat=X_train_transform.shape[1], n_parents=640, mutation_rate=0.20, n_gen=2,
                                     X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train, Y_test=y_test)
# selectedChromo = np.empty(X_train_transform.shape[1])
for chromo in chromo_df_bc:
    numOfSelectedFeatures = np.sum(chromo)
    # if numOfSelectedFeatures > max:
    # selectedChromo = chromo
    print("number of features in chromo", numOfSelectedFeatures)
new_X_train_transform = X_train_transform[:, chromo_df_bc[1]]
JM_flat_data, dataMean = JM_flat(new_X_train_transform, y_train)
avg_jm = np.mean(JM_flat_data, axis=1)
with open("avg_jm", 'wb') as file:
    pickle.dump(avg_jm, file)

algo_dict = {0: "fisher", 1: "mrmr", 2: "relief", 3: "dm", 4: "random"}
num_features = []
for i in range(10, 201, 10):
    num_features.append(i)
    fishers_selected = fisher_ranking(new_X_train_transform, y_train, i)
    mrmr_selected = mrmr_ranking(new_X_train_transform, y_train, i)
    relief_selected = relieff_ranking(new_X_train_transform, y_train, i)
    dm_selected, dm_coordinates = dm_ranking(JM_flat_data, i, 80)
    with open("dm_coordinates", 'wb') as file:
        pickle.dump(dm_coordinates, file)
    random_selected = random_choosing(i)
    for j in range(5):
        features_dict = {0: fishers_selected, 1: mrmr_selected,
             2: relief_selected, 3: dm_selected, 4: random_selected}
        classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier_selected.fit(
            X_train_transform[:, features_dict[j]], y_train)
        predictions = classifier_selected.score(
            X_test_transform[:, features_dict[j]], y_test)
        with open(f'{algo_dict[j]}_ga', 'wb') as file:
            pickle.dump(predictions, file)
        with open(f'{algo_dict[j]}_selected', 'wb') as file:
            pickle.dump(algo_dict[j], file)

