""" This file loads preprocessed dataset as a pandas Dataframe and select the Train & Test corpus"""

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(dataset_path, category_feature, feature_to_drop=None):
    """
    This function loads the Features csv file into a Pandas Dataframe
    and creates the Train and Test split with it.
    If the parameter :param feature_to_drop is specified, it creates those
    without the specified feature.
    :param dataset_path: path of the csv Features file
    :param category_feature: name of the label column
    :param feature_to_drop: name of the column to drop for the ablation study
    :return: Train Features, Test Features, Train Labels and Test Labels
    """
    dataset_df = load_csv(dataset_path)
    y = set_labels(dataset_df)
    if feature_to_drop:
        X = set_features(dataset_df, category_feature, feature_to_drop)
    else:
        X = set_features(dataset_df, category_feature)
    X_train, X_test, y_train, y_test = split_dataset_into_test_train(X, y)
    return X_train, X_test, y_train, y_test


def load_csv(dataset_path, delimiter=";"):
    dataset_df = pd.read_csv(dataset_path, delimiter=delimiter)
    return dataset_df


def set_labels(dataset_df, category_feature="opportunity_stage_after_30_days"):
    # Putting labels variable to y
    y = dataset_df[category_feature]
    print(y.value_counts())
    # 1 = Prospects 0 = Clients
    return y


def set_features(dataset_df, category_feature="opportunity_stage_after_30_days", feature_to_remove=None):
    # Putting feature variable to X
    if feature_to_remove:
        to_drop = [feature_to_remove, category_feature]
    else:
        to_drop = category_feature
    X = dataset_df.drop(to_drop, axis=1)
    return X


def set_all_features(dataset_df, category_feature="opportunity_stage_after_30_days"):
    return dataset_df.drop(category_feature, axis=1)


def split_dataset_into_test_train(X, y, test_size=0.20, random_state=42):
    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=test_size,
                                                        random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test
