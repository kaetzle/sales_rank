import sys
import time

from tqdm import tqdm

import training as tr
import datasets as ds
import evaluation as eval
import ablation_study as ab


def train(dataset_path, classifier_type, category_feature):
    X_train, X_test, y_train, y_test = ds.prepare_data(dataset_path, category_feature, feature_to_drop=None)
    classifier = tr.train_model(X_train, y_train, classifier_type)
    eval.compute_evaluation_report(classifier, X_test, y_test)


def run_ablation_study(dataset_path, classifier_type,category_feature, feature_to_drop):
    ab.run_ablation_study(dataset_path, classifier_type,feature_to_drop, category_feature)


if __name__ == '__main__':
    start_time = time.time()
    if len(sys.argv) >= 2:
        mode = sys.argv[1]
        if mode == "--train":
            if len(sys.argv) == 5:
                train(sys.argv[2], sys.argv[3], sys.argv[4])
            else:
                print(
                    "USAGE : python3 __main__py --train <path to the csv file> <LogisticRegression or RandomForest> "
                    "<Name of the labels columns>")
        if mode == "--ablation":
            if len(sys.argv) == 6:
                run_ablation_study(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
            else:
                print(
                    "USAGE : python3 __main__py --ablation <path to the csv file> <LogisticRegression or RandomForest> "
                    "<Name of the labels columns> <Name of the feature to remove for the ablation study>")
    else:
        print("USAGE : python3 __main__py <mode: --train --ablation> <path to the csv file> ...")
    print("--- %s seconds ---" % (time.time() - start_time))

