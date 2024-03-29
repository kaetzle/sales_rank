""" This file run a small ablation study for one specified Feature at a time"""

import datasets as ds
import training as tr
import evaluation as eval


def run_ablation_study(dataset_path, classifier_type,
                       feature_to_drop, category_feature="opportunity_stage_after_30_days"):
    """
    This functional is used to run a comparison between 2 classifiers:
    One with all the Features in the dataset file,
    the other by dropping the column name
    indicated by feature_to_drop in order to run a small ablation study.
    :param dataset_path: path of the csv Features file
    :param classifier_type: for now "LogisticRegression" or "RandomForest"
    :param category_feature: name of the label column
    :param feature_to_drop: name of the column to drop for the ablation study
    :return: report of the model with all Features and
    report of the model without the dropped Feature
    """
    #Get results for the baseline classifier
    X_train, X_test, y_train, y_test = ds.prepare_data(dataset_path, category_feature)
    classifier = tr.train_model(X_train, y_train, classifier_type)

    #Get results for the classifier with the dropped features
    X_train_drop, X_test_drop, y_train_drop, y_test_drop = ds.prepare_data(dataset_path,
                                                                           category_feature, feature_to_drop)
    drop_classifier = tr.train_model(X_train_drop, y_train_drop, classifier_type)
    all_report = eval.compute_evaluation_report(classifier, X_test, y_test)
    ablation_report = eval.compute_evaluation_report(drop_classifier, X_test_drop, y_test_drop)
    return all_report, ablation_report
