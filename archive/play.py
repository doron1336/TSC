from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import optim
from models.minirocket import fit, transform
from sktime.datasets import load_osuleaf
from utils.JM import JM_flat
from torch.utils.data import DataLoader
from models.VAE import Autoencoder, weights_init_uniform_rule, customLoss, DataBuilder, DataBuilderRocket

x_train, y_train = load_osuleaf(split="train", return_X_y=True)
x_test, y_test = load_osuleaf(split="test", return_X_y=True)

# ment to normalize the class names (start at 0 always)
y_test = [int(i) for i in y_test]
if np.min(y_test) != 0:
    y_test = [str(int(i)-1) for i in y_test]
    y_train = [str(int(i)-1) for i in y_train]

# labels, counts = np.unique(y_train, return_counts=True)
# print(labels, counts)

parameters = fit(x_train)
X_train_transform = transform(x_train, parameters)

data, dataMean = JM_flat(X_train_transform, y_train)


vae_results = torch.load('firstTry_latent5.pt')
print(vae_results.shape)
zdata = vae_results[:, 0].cpu().numpy()
xdata = vae_results[:, 1].cpu().numpy()
ydata = vae_results[:, 2].cpu().numpy()
ax = plt.axes(projection="3d")

ax.scatter(xdata, ydata, zdata, c=dataMean, cmap='Blues')
plt.show()
