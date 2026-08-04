from matplotlib import pyplot as plt
import torch
import numpy as np
from torch import optim
from models.minirocket import fit, transform
from sktime.datasets import load_osuleaf
from utils.JM import JM_flat
from torch.utils.data import DataLoader
from models.VAE import Autoencoder, weights_init_uniform_rule, customLoss, DataBuilder, DataBuilderRocket

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

data_set = DataBuilderRocket(data)
trainloader = DataLoader(dataset=data_set, batch_size=1024)
# print(type(trainloader.dataset.x))
# print(data_set.x.shape)


D_in = data_set.x.shape[1]
H = 50
H2 = 12
latent_dim = 2
model = Autoencoder(D_in, H, H2, latent_dim).to(device)
model.apply(weights_init_uniform_rule)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_mse = customLoss()

# Train Model
epochs = 2500
log_interval = 50
val_losses = []
train_losses = []


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(trainloader):
        data = data.to(device)
        data = data.to(torch.float32)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_mse(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
#        if batch_idx % log_interval == 0:
#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                epoch, batch_idx * len(data), len(trainloader.dataset),
#                       100. * batch_idx / len(trainloader),
#                       loss.item() / len(data)))
    if epoch % 200 == 0:
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(trainloader.dataset)))
        train_losses.append(train_loss / len(trainloader.dataset))


for epoch in range(1, epochs + 1):
    train(epoch)

## Evaluate ##

model.eval()
test_loss = 0
# # no_grad() bedeutet wir nehmen die vorher berechneten Gewichte und erneuern sie nicht
with torch.no_grad():
    for i, data in enumerate(trainloader):
        data = data.to(device)
        data = data.to(torch.float32)
        recon_batch, mu, logvar = model(data)


# Get Embeddings

mu_output = []
logvar_output = []

with torch.no_grad():
    for i, (data) in enumerate(trainloader):
        data = data.to(device)
        data = data.to(torch.float32)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        mu_tensor = mu
        mu_output.append(mu_tensor)
        mu_result = torch.cat(mu_output, dim=0)

        logvar_tensor = logvar
        logvar_output.append(logvar_tensor)
        logvar_result = torch.cat(logvar_output, dim=0)

print(mu_result.shape)
# torch.save(mu_result, 'firstTry_latent5.pt')
# ax = plt.axes(projection='3d')


# vae_results = torch.load('firstTry_latent5.pt')
vae_results = mu_result
zdata = vae_results[:, 0].cpu().numpy()
xdata = vae_results[:, 1].cpu().numpy()
# ydata = vae_results[:, 2].cpu().numpy()
ax = plt.axes(projection="3d")

ax.scatter(xdata, zdata, c=dataMean, cmap='Blues')
plt.show()

# Data for three-dimensional scattered points
# zdata = mu_result[:, 0].cpu().numpy()
# xdata = mu_result[:, 1].cpu().numpy()
# ydata = mu_result[:, 2].cpu().numpy()
# ax.scatter(xdata, ydata, zdata)
# plt.show()
