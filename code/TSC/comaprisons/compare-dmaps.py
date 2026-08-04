import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3  # noqa: F401
import numpy as np
from sklearn.datasets import make_s_curve
import os
import pickle
import datafold.dynfold as dfold
import datafold.pcfold as pfold
from datafold.dynfold import LocalRegressionSelection
from datafold.utils.plot import plot_pairwise_eigenvector
from TSC.diffusionMaps.Diffusion_Maps import diffusionMapping
from TSC.utils.JM import JM_flat
from TSC.models.GA import generations
import paths  # noqa: F401  # TSC path config

os.chdir(os.path.join(paths.HAND_MOVEMENT_DIR, "Database", "handmovement3", "1"))
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

rng = np.random.default_rng(1)
nr_samples = len(new_X_train_transform[1])
# reduce number of points for plotting
nr_samples_plot = 1000
idx_plot = rng.permutation(nr_samples)[0:nr_samples_plot]
avg_jm = np.mean(JM_flat_data, axis=1)

# Optimize kernel parameters
X_pcm = pfold.PCManifold(JM_flat_data)
X_pcm.optimize_parameters(result_scaling=3)

print(f"epsilon={X_pcm.kernel.epsilon}, cut-off={X_pcm.cut_off}")

# datafold Diffusion Maps
dmap = dfold.DiffusionMaps(
    kernel=pfold.GaussianKernel(
        epsilon=X_pcm.kernel.epsilon, distance=dict(cut_off=X_pcm.cut_off)
    ),
    n_eigenpairs=9,
)
dmap = dmap.fit(X_pcm)
evecs, evals = dmap.eigenvectors_, dmap.eigenvalues_
# coordinates_ = evecs[:,[1,2]]
# print(coordinates_.shape)
# exit()

plot_pairwise_eigenvector(
    eigenvectors=dmap.eigenvectors_[idx_plot, :],
    n=1,
    fig_params=dict(figsize=[15, 15]),
    scatter_params=dict(cmap=plt.cm.Spectral, c=avg_jm[idx_plot]),
)


# Our diffusionMaps

eps_type = 'mean'  # mean' #or maxmin
alpha = 1
vecs, eigs, coordinates, dataList, epsilon = diffusionMapping(
    JM_flat_data, alpha, eps_type, 1, dim=5)  # dim - number of diffusion coordinates computed
print(coordinates.shape)
fig, ax = plt.subplots()
fig = plt.figure(figsize=(12, 12))

plot_pairwise_eigenvector(
    eigenvectors=coordinates[idx_plot, :],
    n=1,
    fig_params=dict(figsize=[15, 15]),
    scatter_params=dict(cmap=plt.cm.Spectral, c=avg_jm[idx_plot]),
)
# sc = ax.scatter(sequence_containing_x_vals,
#                 sequence_containing_y_vals, c=X_color[idx_plot], cmap=plt.cm.Spectral)
# plt.colorbar(sc)
plt.show()