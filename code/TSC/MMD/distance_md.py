import os

import numpy as np
from models.minirocket import fit, transform

from utils.retrieve_minirocket import retrieve_minirocket_data
from sklearn.metrics.pairwise import rbf_kernel

baseDir = "UCRArchive_2018"
os.chdir(os.path.join(os.path.dirname(os.getcwd()), baseDir))
dataset_name: str = 'Haptics'
DIRECTORY_NUM: int = 4

def wtype(data):
	return type(data).__name__

def ensure_numpy(data, rounding=None, ensure_column_format=True):

	if wtype(data) == 'ndarray':
		if ensure_column_format and len(data.shape) == 1:
			X = np.atleast_2d(data).T
		else:
			X = data
	elif wtype(data) == 'Index':
		return data.to_numpy()
	elif wtype(data) == 'list':
		return np.array(data)
	elif wtype(data) == 'wData':
		X = data.df.values
	elif wtype(data) == 'DataFrame':
		X = data.values
	elif wtype(data) == 'Tensor':
		X = data.detach().cpu().numpy()
	elif wtype(data) == 'Series':
		X = data.values
	elif wtype(data) == 'Int64Index':
		return data.to_numpy()
	elif np.isscalar(data):
		X = np.array([[data]])
	elif wtype(data) == 'dimension_reduction':
		X = data.Ӽ
	else:
		raise ValueError('Unknown dataType %s'%wtype(data))

	if rounding is not None: X = np.round(X, rounding)
	return X


def mmd(X, Y, sigma=1):
	X = np.squeeze(ensure_numpy(X))
	Y = np.squeeze(ensure_numpy(Y))

	n = X.shape[0]
	m = Y.shape[0]
	gamma = 1.0 / (2 * sigma * sigma)

	Kx = np.sum(rbf_kernel(X, gamma=gamma))
	Ky = np.sum(rbf_kernel(Y, gamma=gamma))
	Kxy = np.sum(rbf_kernel(X, Y, gamma=gamma))

	mmd_squared = Kx / (n * n) - Kxy * 2 / (m * n) + Ky / (m * m)
	return mmd_squared

def sub_vec(vector, num_to_keep:int):
	# Convert to a numpy array for easy manipulation
	vector = np.array(vector)

	# Create a mask for the elements to keep
	mask = np.zeros(vector.shape[1], dtype=bool)
	mask[:num_to_keep] = True

	# Apply the mask to columns
	result = vector * mask

	return result


def trim_and_rocket(x, number_of_features):
	trimmed_x_train = trim_vector(x, number_of_features)
	parameters = fit(trimmed_x_train)
	X_train_transform_trimmed = transform(trimmed_x_train, parameters)
	return X_train_transform_trimmed

def trim_vector(vector, n: int):
	if not isinstance(vector, np.ndarray):
		raise ValueError("Input must be a NumPy array.")
	if vector.shape[1] < n:
		raise ValueError("The input vector has fewer columns than n.")
	return vector[:, :n]

directory = os.path.join(os.path.abspath("."), dataset_name, str(DIRECTORY_NUM))
print(directory)
X_train_transform, X_test_transform, y_train, y_test = retrieve_minirocket_data(directory, dataset_name)
