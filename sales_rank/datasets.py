import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset:
    """ This class allows you to load preprocessed dataset as a pandas Dataframe"""

    def load_csv(self, dataset_path, delimiter=";"):
        dataset_df = pd.read_csv(dataset_path, delimiter=delimiter)
        return dataset_df

    def split_dataset_into_test_train(self, dataset_df, category_feature="opportunity_stage_after_30_days",
                                      test_size="0.2", random_state=42):
        # Putting feature variable to X
        X = dataset_df.drop(category_feature, axis=1)
        # Putting response variable to y
        y = dataset_df[category_feature]
        print(y.value_counts())
        # 1 = Prospects 0 = Clients
        # Splitting the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=test_size,
                                                            random_state=random_state, stratify=y)
        return X_train, X_test, y_train, y_test
