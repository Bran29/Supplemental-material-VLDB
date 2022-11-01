import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class clust:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = (
            x_train,
            x_test,
            y_train,
            y_test,
        )

    def classify(self, model=LogisticRegression(random_state=42)):
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        print("Accuracy: {}".format(accuracy_score(self.y_test, y_pred)))

    def Kmeans(self, output="add"):
        n_clusters = len(np.unique(self.y_train))
        clf = KMeans(n_clusters=n_clusters, random_state=42)
        clf.fit(self.X_train)
        # y_labels_train = clf.labels_
        y_labels_test = clf.predict(self.X_test)
        # print("Accuracy: {}".format(accuracy_score(self.y_test, y_labels_test)))
        return accuracy_score(self.y_test, y_labels_test)
        # if output == "add":
        #     self.X_train["km_clust"] = y_labels_train
        #     self.X_test["km_clust"] = y_labels_test
        # elif output == "replace":
        #     self.X_train = y_labels_train[:, np.newaxis]
        #     self.X_test = y_labels_test[:, np.newaxis]
        # else:
        #     raise ValueError("output should be either add or replace")
        # return self
