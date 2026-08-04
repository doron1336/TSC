from cProfile import label
# from tslearn.datasets import UCR_UEA_datasets
# from sktime.transformations.panel.rocket import MiniRocket
from enum import unique
from re import T
import numpy as np
from utils.JM import *
from sktime.utils.validation.panel import check_X
# from sktime.transformations.panel.rocket import Rocket
from models.minirocket import fit, transform
from models.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV
from sktime.datasets import load_arrow_head, load_osuleaf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from iterativeProcess import * 


# from IPython.display import display
# x_test, y_test, x_train, y_train = UCR_UEA_datasets().load_dataset("ArrowHead")

# x_train, y_train = load_arrow_head(split="test", return_X_y=True)
# x_test, y_test = load_arrow_head(split="train", return_X_y=True)

x_train, y_train = load_osuleaf(split="train", return_X_y=True)
x_test, y_test = load_osuleaf(split="test", return_X_y=True)

y_test = [int(i) for i in y_test] ## ment to normalize the class names (start at 0 always)
if np.min(y_test) != 0:
    y_test = [str(int(i)-1) for i in y_test]
    y_train = [str(int(i)-1) for i in y_train]

# labels, counts = np.unique(y_train, return_counts=True)
# print(labels, counts)
parameters = fit(x_train)
X_train_transform = transform(x_train, parameters)

# np.save('Osuleaf_MiniRocket.npy', X_train_transform)
# with open('./Osuleaf_MiniRocket.npy', 'rb') as f:
    # X_train_transform = np.load(f)
# X_train_transform = X_train_transform.to_numpy(dtype ='float32')
# print(type(X_train_transform))
# classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# classifier.fit(X_train_transform, y_train)

# X_test_transform = transform(x_test, parameters)

# predictions = classifier.score(X_test_transform, y_test)
# print(predictions)

# with open('./arrowHead_miniRocket.npy', 'rb') as f:
    # X_train_transform = np.load(f)

# print(X_train_transform.shape)
# example = X_train_transform[0,0:1000]
# print(example)
# matrix = np.zeros([3,3])
# for i in range(3):
#     loc = np.where(y_train == str(i))
#     for j in range(3):
#         loc2 = np.where(y_train == str(j))
#         if i == j:
#             matrix[i][j] = 0
#         else:
#             a = np.asarray(example)
#             matrix[i][j] = computeJM(np.asarray(a[loc]), np.asarray(a[loc2]), 143)
# print(matrix)
# print(X_train_transform.shape)
# print(type(X_train_transform))
# print(np.quantile(meanJMVec, q=np.arange(0.1,1,0.1)))
# newVec = meanJMVec[meanJMVec > 0.25839496]

dictionatyJM = {}

meanJMVec, maxJMVec, dictionatyJM = generateJMVector(X_train_transform, y_train)
print(len(dictionatyJM[0]))
indicesOfRandomFreatures = np.random.randint(9996, size=(10)) 
np.save('indicesOfRandomFreatures.npy', np.asarray(indicesOfRandomFreatures))
k = np.linspace(0.2, 0.7, 20)
# k=[1]
scores = []
numOfSamples1 = []
for m in range(len(k)):
    indicesOfImportantFreatures = [i for i in range(len(meanJMVec)) if meanJMVec[i] > 0.3]
    # with open('./indicesOfRandomFreatures.npy', 'rb') as f:
        # indicesOfImportantFreatures = np.load(f)
    if len(indicesOfImportantFreatures) != 0:
        numOfSamples1.append(len(indicesOfImportantFreatures))
        newX_train_transform = X_train_transform[:,indicesOfImportantFreatures]
        print(newX_train_transform.shape)
        print(f"The number of features {len(indicesOfImportantFreatures)}")

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(newX_train_transform, y_train)
        X_test_transform = transform(x_test, parameters)
        predictions = classifier.predict(X_test_transform[:,indicesOfImportantFreatures])
        # X_test_transform = X_test_transform.to_numpy(dtype ='float32')
        scoress = classifier.score(X_test_transform[:,indicesOfImportantFreatures], y_test)
        # print(predictions)
        # print(y_test)
        cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
        disp.plot()
        # plt.show()
        target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
        print(classification_report(y_test, predictions, target_names=target_names))
        scores.append(scoress)
        print(scoress)

# np.save('predictionScore.npy', np.asarray(scores))
# np.save('numOfSamples.npy', np.asarray(numOfSamples1))
# with open('./predictionScore.npy', 'rb') as f:
#     scores = np.load(f)
# with open('./numOfSamples.npy', 'rb') as f:
#     numOfSamples1 = np.load(f)

# k2 = np.linspace(0.2, 2, 40)
# maxScores = []
# numOfSamples = []
# for m in range(len(k2)):
#     indicesOfImportantFreatures = [i for i in range(len(maxJMVec)) if maxJMVec[i] > k2[len(k2)-m-1]]
 
#     if len(indicesOfImportantFreatures) != 0:
#         numOfSamples.append(len(indicesOfImportantFreatures))
#         newX_train_transform = X_train_transform[:,indicesOfImportantFreatures]

#         classifier2 = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
#         print(f"The number of features {len(indicesOfImportantFreatures)}")
#         classifier2.fit(newX_train_transform, y_train)
#         X_test_transform = transform(x_test, parameters)
#         # X_test_transform = X_test_transform.to_numpy(dtype ='float32')
#         predictions = classifier2.score(X_test_transform[:,indicesOfImportantFreatures], y_test)
#         maxScores.append(predictions)
#         print(predictions)

# classifier3 = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
# classifier3.fit(X_train_transform, y_train)
# X_test_transform = transform(x_test, parameters)
# # X_test_transform = X_test_transform.to_numpy(dtype ='float32')
# print(X_test_transform.shape)
# predictions = classifier3.score(X_test_transform, y_test)

# print(f"original series score {predictions}")

# fig, ax1 = plt.subplots(1, 1)
# ax1.plot(numOfSamples1, scores, label="mean")
# ax1.plot(numOfSamples, maxScores, label="max")
# ax1.plot(predictions*np.ones(10000), label="original series score", color='red')
# ax1.set(xlabel='num Of Samples', ylabel='prediction score')
# ax1.legend(loc = 'lower right')

# plt.show()


# x_train = check_X(x_train, enforce_univariate=True, coerce_to_numpy=True)
# x_train = x_train[:, 0, :].astype(np.float32)
# classifier2.fit(X_train_transform, y_train)
# x_test = check_X(x_test, enforce_univariate=True, coerce_to_numpy=True)
# x_test = x_test[:, 0, :].astype(np.float32)
# predictions = classifier2.score(x_test, y_test)
# print(f"original series score {predictions}")

# Rocket

# rocket = Rocket()  # by default, ROCKET uses 10,000 kernels
# rocket.fit(x_train)
# X_train_transform = rocket.transform(x_train)

# classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
# classifier.fit(X_train_transform, y_train)

# X_test_transform = rocket.transform(x_test)

# predictions = classifier.score(X_test_transform, y_test)
# print(predictions)


# compute cosine similarity
cosine = 0
counter1, counter2, counter3 = 0, 0, 0
class1, class2, class3 = [], [], []

# for i in range(len(X_train_transform)):
    # if y_train[i] == '0' :
        # class1.append(X_train_transform[i])
        # counter1 += 1
    # if y_train[i] == '1' :
        # class2.append(X_traicn_transform[i])
        # counter2 += 1
    # if y_train[i] == '2' :
        # class3.append(X_train_transform[i])
        # counter3 += 1

# pair_order_list = itertools.permutations(list, 2)

# for j in pair_order_list:
#     counter += 1
#     cosine += distance.cosine(j[0], j[1])
#     dist = np.linalg.norm(j[0] - j[1])

# print("Cosine Similarity:", cosine/counter)
# print("Euclidean distance Similarity:", dist/counter)
# fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
# for i in range(6,10):
    # fig, ax1 = plt.subplots(1, 1)
    # class1Sorted = np.sort(class1[i])
    # # ax1.plot(class1[0:500])
    # ax1.plot(class1Sorted, label="class1")
    # ax1.set(xlabel='sample', ylabel='amplitude')
    # ax1.set_title('class 1')
    # plt.subplot(2, 1, 1)
    # plt.plot(X_train_transform[3])

    # order = class1[i].argsort()

    # class2Sorted = class2[i][order]
    # ax2.plot(class2[0:500])
    # ax1.plot(class2Sorted, label="class 2 by class1 sort")
    # ax1.set(xlabel='sample', ylabel='amplitude')
    # ax1.set_title('class 2')
    # plt.subplot(2, 1, 2)
    # plt.plot(X_train_transform[60])

    # class3Sorted = class3[i][order]
    # ax3.plot(class3[0:500])
    # ax1.plot(class3Sorted, label="class 3 by class1 sort")
    # ax1.set(xlabel='sample', ylabel='amplitude')
    # ax1.set_title('class 3')
    # plt.subplot(2, 1, 3)
    # plt.plot(X_train_transform[3])

    # ax1.legend(loc="upper left")
    # ax1.set_title("Two samples of same class")
    # plt.show()
    # plt.savefig(f"{i}_example_same2")