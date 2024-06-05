import os
from models.minirocket import fit, transform
from handmovementDATA_randomTest import GetHandMovementDATA_randomTest
from sktime.datasets import load_arrow_head, load_osuleaf
import pickle
import numpy as np

baseDir = "handMovement/Database"
os.chdir(baseDir)
print("Current working directory:", os.getcwd())

# load DATA
x_train, y_train, x_test, y_test = GetHandMovementDATA_randomTest(precentOfTest=0.15, directory=os.getcwd())
for num in range(1, 6):
    directory = f"handmovement5/{num}"
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            print("failed to create directory")

    # miniRocket section
    filename_train = os.path.join(os.path.abspath("."), directory, "handmovement_minirocket_train")
    filename_test = os.path.join(os.path.abspath("."), directory, "handmovement_minirocket_test")
    filename_y_train = os.path.join(os.path.abspath("."), directory, "handmovement_y_train")
    filename_y_test = os.path.join(os.path.abspath("."), directory, "handmovement_y_test")

    parameters = fit(x_train.to_numpy(dtype=np.float32))
    X_train_transform = transform(x_train.to_numpy(dtype=np.float32), parameters)
    X_test_transform = transform(x_test.to_numpy(dtype=np.float32), parameters)


    # data, dataMean = JM_flat(X_train_transform, y_train)
    # print("size of data from JM_flat", data.shape)

    with open(filename_train, 'wb') as file:
        pickle.dump(X_train_transform, file)
    with open(filename_test, 'wb') as file:
        pickle.dump(X_test_transform, file)

    with open(filename_y_train, 'wb') as file:
        pickle.dump(y_train, file)
    with open(filename_y_test, 'wb') as file:
        pickle.dump(y_test, file)

# with open(filename_train, 'rb') as file:
#     X_train_transform = pickle.load(file)
# with open(filename_test, 'rb') as file:
#     X_test_transform = pickle.load(file)
# with open("arrowhead_y_train", 'rb') as file:
#     y_train = pickle.load(file)
# with open("arrowhead_y_test", 'rb') as file:
#     y_test = pickle.load(file)
# with open(filename_train, 'wb') as f:
#     np.save(f, X_train_transform)
# with open(filename_train, 'wb') as f:
#     np.save(f, X_test_transform)

# with open(filename_train, 'rb') as f:
#     X_train_transform = np.load(f)
# with open(filename_train, 'rb') as f:
#     X_test_transform = np.load(f)

# print(X_train_transform.shape)
# classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

# classifier.fit(X_train_transform, y_train)
# predictions = classifier.score(X_test_transform, y_test)
# print(predictions)
