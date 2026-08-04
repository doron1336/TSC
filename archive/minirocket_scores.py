# Here i should go through the results which is saved for every dataset - take those results and find which
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV

from utils.retrieve_minirocket import retrieve_minirocket_data
from utils.topK_indices import calc_score_rf

df = pd.DataFrame(columns=['Dataset', 'Score_miniRocket'])

baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.getcwd(), baseDir))
# List the contents of the directory with full paths
datasets = [os.path.join(os.getcwd(), item) for item in
            os.listdir(os.path.join(os.getcwd()))]

problematic = ['Wafer', 'ECGFiveDays', 'GunPointAgeSpan', 'BeetleFly', 'Adiac', 'ItalyPowerDemand', 'Yoga', 'GunPointOldVersusYoung',
               '.DS_Store', "Missing_value_and_variable_length_datasets_adjusted", 'timer.log']
dataset_names, scores = [], []
for i in datasets:
    dataset_name = i.split('/')[-1]
    print(dataset_name)
    target_subfolder = "4"
    try:
        # Load the Data - miniRocket section
        x_train_transform, x_test_transform, y_train, y_test = retrieve_minirocket_data(
            os.path.join(i, target_subfolder), dataset_name)
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(x_train_transform, y_train)
        score_rf = calc_score_rf(x_train_transform, x_test_transform, y_train, y_test)
        score = classifier.score(x_test_transform, y_test)
        dataset_names.append(dataset_name)
        scores.append(score)
    except Exception as e:
        print(f"not found for {dataset_name}")

rows = []
for dataset_name, score in zip(dataset_names, scores):
    rows.append({'Dataset': dataset_name, 'Score_miniRocket': score})
df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df.to_csv('/Users/doron/Desktop/personal/thesis/TSC/datasets_scores_4.csv', index=False)
