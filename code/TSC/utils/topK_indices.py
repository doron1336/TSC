import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV


def calc_score(x_train_transform, x_test_transform, y_train, y_test):
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(x_train_transform, y_train)
    return classifier.score(x_test_transform, y_test)


def calc_score_rf(x_train_transform, x_test_transform, y_train, y_test):
    classifier = RandomForestClassifier(
        n_estimators=100,  # Number of trees
        max_features='sqrt',  # Suitable for high-dimensional data
        random_state=42,  # For reproducibility
        n_jobs=-1  # Use all available cores for faster training
    )
    classifier.fit(x_train_transform, y_train)
    return classifier.score(x_test_transform, y_test)
