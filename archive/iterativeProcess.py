from cProfile import label
# from tslearn.datasets import UCR_UEA_datasets
# from sktime.transformations.panel.rocket import MiniRocket
from enum import unique
from re import T
import numpy as np
from utils.JM import *
from utils.more_features import * 
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

x_train, y_train = load_osuleaf(split="train", return_X_y=True)
x_test, y_test = load_osuleaf(split="test", return_X_y=True)

y_test = [int(i) for i in y_test] ## ment to normalize the class names (start at 0 always)
if np.min(y_test) != 0:
    y_test = [str(int(i)-1) for i in y_test]
    y_train = [str(int(i)-1) for i in y_train]

parameters = fit(x_train)
X_train_transform = transform(x_train, parameters)
dictionatyJM = {}

meanJMVec, maxJMVec, dictionatyJM = generateJMVector(X_train_transform, y_train)
numOfClasses = len(np.unique(y_train))
indicesOfRandomFreatures = np.random.randint(9996, size=(10)) 

scoreGOAL = 0.96
score = 0
indicesOfImportantFreatures = indicesOfRandomFreatures

while score < scoreGOAL:
    print(f"The number of features {len(indicesOfImportantFreatures)}")

    newX_train_transform = X_train_transform[:,indicesOfImportantFreatures]

    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(newX_train_transform, y_train)
    X_test_transform = transform(x_test, parameters)
    predictions = classifier.predict(X_test_transform[:,indicesOfImportantFreatures])
    # X_test_transform = X_test_transform.to_numpy(dtype ='float32')
    scoress = classifier.score(X_test_transform[:,indicesOfImportantFreatures], y_test)

    cm = confusion_matrix(y_test, predictions, labels=classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
    disp.plot()
    # plt.show()
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4', 'class 5']
    np.fill_diagonal(cm, 0) # the main diagonal of the confusion matrix is not interesting 
    max_array = cm.max(1)
    indices = cm.argmax(1)
    prob_class = max_array.argmax()
    sec_prob_class = indices[prob_class]
    more_feature_indices = get_more_features(dictionatyJM, prob_class, sec_prob_class, numOfClasses, 10)

    translateIndices = [more_feature_indices[i]-numOfClasses*i for i in range(len(more_feature_indices))]
    for i in translateIndices:
        dictionatyJM[prob_class][i] = 0 # zero out features we already picked

    print(classification_report(y_test, predictions, target_names=target_names))
    print(scoress)
    
    score = scoress
    indicesOfImportantFreatures = np.append(indicesOfImportantFreatures, more_feature_indices)

