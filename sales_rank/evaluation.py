import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def compute_evaluation_report(classifier, X_test, y_test):
    y_pred, proba = compute_test_classifier(classifier, X_test)
    auc_score = calcultate_auc_score(classifier, X_test, y_test)
    report_f1 = calculate_f1_metrics(y_test, y_pred)
    report = "For this model : \n" \
             + "The AUC score is : " + str(auc_score) + "\n" + report_f1
    print(report)
    conf_matrix = plot_confusion_matrix(y_test, y_pred)


def compute_test_classifier(classifier, X_test):
    y_pred = classifier.predict(X_test)
    proba = classifier.predict_proba(X_test)
    return y_pred, proba


def calcultate_auc_score(classifier, X_test, y_test):
    return roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1])


def plot_confusion_matrix(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    conf_matrix = plt.figure(figsize=(8, 4))
    sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Blues, linewidths=0.2)
    # add labels to plot
    class_names = ["Not Customer", "Customer"]
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5
    plt.xticks(tick_marks, class_names, rotation=25)
    plt.yticks(tick_marks2, class_names, rotation=0)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')
    plt.show()
    return conf_matrix


def calculate_f1_metrics(y_test, y_pred):
    return classification_report(y_test, y_pred)
