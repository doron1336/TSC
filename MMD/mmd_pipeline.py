# 1/ pick a dataset
# 2/ use minirocket to create the transform
# 3/ calculate the JM - take the flattened version
# 4/ same process with shorter signal
# 5/ calculate gmm
import os
import pickle

from matplotlib import pyplot as plt

from MMD.distance_md import mmd, trim_and_rocket
from utils.JM import JM_flat
from utils.kmeans import perform_kmeans_clustering
from utils.retrieve_minirocket import retrieve_minirocket_data, load_ucr_dataset
from utils.file_system import save_to_pickle
from utils.topK_indices import calc_score

baseDir = "UCRArchive_2018"
without_GA = 'without_GA'

os.chdir(os.path.join(os.path.dirname(os.getcwd()), baseDir))
dataset_name: str = 'GestureMidAirD2'
DIRECTORY_NUM: int = 4
base_dir = os.path.join(os.path.abspath("."), dataset_name)
directory = os.path.join(base_dir, str(DIRECTORY_NUM))
print(directory)

X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)
prediction_score = calc_score(X_train_transform, X_test_transform, y_train, y_test)
print(prediction_score)
if os.path.exists(os.path.join(directory, without_GA, 'JM_flat_data_full')):
    with open(os.path.join(directory, without_GA, 'JM_flat_data_full'), 'rb') as file:
        JM_flat_data = pickle.load(file)
else:
    JM_flat_data, _ = JM_flat(X_train_transform, y_train)
x_train, y_train, x_test, y_test = load_ucr_dataset(base_dir)


x_train_range = range(50, x_train.shape[1]+1, 50)

mmd_result = []
for i in x_train_range:
    print(i)
    X_train_transform_trimmed = trim_and_rocket(x_train, i)
    X_test_transform_trimmed = trim_and_rocket(x_test, i)
    prediction_score = calc_score(X_train_transform_trimmed, X_test_transform_trimmed, y_train, y_test)
    # mmd_result.append(mmd(X_train_transform, X_train_transform_trimmed))
    print(prediction_score)
    if os.path.exists(os.path.join(f'/Users/doron/Desktop/personal/thesis/TSC/MMD/{dataset_name}', f"JM_trimmed_{i}")):
        with open(os.path.join(f'/Users/doron/Desktop/personal/thesis/TSC/MMD/{dataset_name}', f"JM_trimmed_{i}"), 'rb') as file:
            JM_flat_data_trimmed = pickle.load(file)
    else:
        JM_flat_data_trimmed, _ = JM_flat(X_train_transform_trimmed, y_train)
        save_to_pickle(JM_flat_data_trimmed, f'/Users/doron/Desktop/personal/thesis/TSC/MMD/{dataset_name}', f"JM_trimmed_{i}")

    mmd_result.append(mmd(JM_flat_data, JM_flat_data_trimmed))


# Example lists

# Create scatter plot
plt.scatter(x_train_range, mmd_result)
plt.xlabel('number of features to keep')
plt.ylabel('MMD score')
plt.title('Scatter Plot')
plt.show()


