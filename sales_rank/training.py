""" This file trains a classifier"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, y_train, classifier_type):
    """
    This function launches the training on a classifier using
    the train specified data.
    :param X_train: Train Features
    :param y_train: Train Labels
    :param classifier_type: LogisticRegression or RandomForest
    :return: trained classifier
    """
    if classifier_type == "RandomForest":
        classifier = RandomForestClassifier(max_depth=20, min_samples_leaf=5, n_estimators=200,
                                           n_jobs=-1, random_state=42, oob_score=True)
        classifier.fit(X_train, y_train)
        return classifier

    if classifier_type == "LogisticRegression":
        classifier = LogisticRegression(random_state=0).fit(X_train, y_train)
        return classifier


def find_parameters_with_gsearch(classifier, X_train, y_train):
    # gridSearch
    params = {
        'max_depth': [2, 3, 5, 10, 20],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'n_estimators': [10, 25, 30, 50, 100, 200]
    }
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=params,
                               cv=4,
                               n_jobs=-1, verbose=1, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
