from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_model(X_train, X_test, y_train, y_test, classifier_type):
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    if classifier_type == "RandomForest":
        classifier_rf = RandomForestClassifier(max_depth=20, min_samples_leaf=5, n_estimators=200,
                                           n_jobs=-1, random_state=42, oob_score=True)
        classifier_rf.fit(X_train, y_train)
        return classifier_rf

    if classifier_type == "LogisticRegression":
        classifier_rf = LogisticRegression(random_state=0).fit(X_train, y_train)
        return classifier_rf


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
