import imp
from sklearn.datasets import (
    load_boston,
    load_iris,
    load_wine,
    load_diabetes,
    load_breast_cancer,
)
import pickle
import pandas as pd


def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df["target"] = pd.Series(sklearn_dataset.target)
    return df


boston_data = load_boston()
df_boston = sklearn_to_df(boston_data)
pickle.dump(df_boston, open("df_boston", "wb"))

iris_data = load_iris()
df_iris = sklearn_to_df(iris_data)
pickle.dump(df_iris, open("df_iris", "wb"))

wine_data = load_wine()
df_wine = sklearn_to_df(wine_data)
pickle.dump(df_wine, open("df_wine", "wb"))

diabetes_data = load_diabetes()
df_diabetes = sklearn_to_df(diabetes_data)
pickle.dump(df_diabetes, open("df_diabetes", "wb"))


breast_cancer_data = load_breast_cancer()
df_breast_cancer = sklearn_to_df(breast_cancer_data)
pickle.dump(df_breast_cancer, open("df_breast_cancer", "wb"))
