import os
import pickle

import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from comparisons_fisher import fisher_ranking
from comparisons_mrmr import mrmr_ranking
from comparisons_relieff import relieff_ranking
from diffusionMaps.Diffusion_Maps import dm_ranking, dm_ranking_datafold
from models.GA import generations
from utils.JM import JM_flat

baseDir = "handMovement/Database"
os.chdir(baseDir)


def random_choosing(num_of_features):
    array_size = 9996
    random_indices = np.random.choice(
        array_size, num_of_features, replace=False)
    # selected_elements = np.zeros(array_size, dtype=bool)
    # selected_elements[random_indices] = True
    return random_indices


for num in range(1, 2):
    directory = os.path.join(os.path.abspath("."), "handmovement8", str(num))
    print("directory ", directory)
    # Load the Data
    dataset_name = "handmovement"
    # miniRocket section
    # # filename_train = os.path.abspath(".") + "\\osuleaf_train"
    # # filename_test = os.path.abspath(".") + "\\osuleaf_test"
    with open(f"{directory}/{dataset_name}_minirocket_train", 'rb') as file:
        X_train_transform = pickle.load(file)
    with open(f"{directory}/{dataset_name}_minirocket_test", 'rb') as file:
        X_test_transform = pickle.load(file)
    with open(f"{directory}/{dataset_name}_y_train", 'rb') as file:
        y_train = pickle.load(file)
    with open(f"{directory}/{dataset_name}_y_test", 'rb') as file:
        y_test = pickle.load(file)
    # with open("JM_FLAT_arrowhead", 'rb') as file:
    # JM_flat_data = pickle.load(file)

    # GA section
    if os.path.exists(f"{directory}/GA_results"):
        with open(f"{directory}/GA_results", 'rb') as file:
            chromo_df_bc, score_bc = pickle.load(file)
    else:
        chromo_df_bc, score_bc = generations(X_train_transform, y_train, size=800, n_feat=X_train_transform.shape[1],
                                             n_parents=640, mutation_rate=0.20, n_gen=2,
                                             X_train=X_train_transform, X_test=X_test_transform, Y_train=y_train,
                                             Y_test=y_test)
    # selectedChromo = np.empty(X_train_transform.shape[1])

    for chromo in chromo_df_bc:
        numOfSelectedFeatures = np.sum(chromo)
        # if numOfSelectedFeatures > max:
        # selectedChromo = chromo
        print("number of features in chromo", numOfSelectedFeatures)
    new_X_train_transform = X_train_transform[:, chromo_df_bc[1]]
    JM_flat_data, dataMean = JM_flat(new_X_train_transform, y_train)
    avg_jm = np.mean(JM_flat_data, axis=1)
    with open(os.path.join(directory,"avg_jm"), 'wb') as file:
        pickle.dump(avg_jm, file)

    algo_dict = {0: "fisher", 1: "mrmr", 2: "relief", 3: "dm", 4: "dm_datafold", 5: "random"}
    predictions_dict = {"fisher": [], "mrmr": [], "relief": [], "dm": [], "dm_datafold": [], "random": []}
    num_algos = len(algo_dict)
    for num_features in range(10, 201, 10):
        fishers_selected = fisher_ranking(new_X_train_transform, y_train, num_features)
        mrmr_selected = mrmr_ranking(new_X_train_transform, y_train, num_features)
        relief_selected = relieff_ranking(new_X_train_transform, y_train, num_features)
        dm_selected, dm_coordinates, labels = dm_ranking(JM_flat_data, num_features, q=100)
        dm_selected_datafold, dm_coordinates_datafold = dm_ranking_datafold(JM_flat_data, num_features, q=100)
        random_selected = random_choosing(num_features)

        features_dict = {0: fishers_selected, 1: mrmr_selected,
                         2: relief_selected, 3: dm_selected, 4: dm_selected_datafold, 5: random_selected}

        dm_data = {"labels": labels, "dm_coordinates": dm_coordinates}
        with open(os.path.join(directory, f"dm_data_{num_features}_features"), 'wb') as file:
            pickle.dump(dm_data, file)
        with open(os.path.join(directory, f"selected_{num_features}_features"), 'wb') as file:
            pickle.dump(features_dict, file)

        for j in range(num_algos):
            classifier_selected = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
            classifier_selected.fit(
                X_train_transform[:, features_dict[j]], y_train)
            prediction_score = classifier_selected.score(
                X_test_transform[:, features_dict[j]], y_test)
            predictions_dict[algo_dict[j]].append(prediction_score)

    for k in range(num_algos):
        with open(os.path.join(directory,f'{algo_dict[k]}_ga_before'), 'wb') as file:
            pickle.dump(predictions_dict[algo_dict[k]], file)
        # with open(f'{algo_dict[k]}_selected', 'wb') as file:
        #     pickle.dump(algo_dict[k], file)
