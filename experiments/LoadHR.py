import pandas as pd
import sys
import os.path as osp
import numpy as np


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


main_dir = osp.dirname(osp.dirname(__file__))

# lib_path = osp.join(this_dir, "..", "OTDD")
add_path(main_dir)

train = pd.read_csv("data/classification2/train.csv")
test = pd.read_csv("data/classification2/test.csv")
# Data Pre-processing
# filling missing values
train["education"].fillna(train["education"].mode()[0], inplace=True)
train["previous_year_rating"].fillna(1, inplace=True)

test["education"].fillna(test["education"].mode()[0], inplace=True)
test["previous_year_rating"].fillna(1, inplace=True)

# removing the employee_id column
train = train.drop(["employee_id", "region"], axis=1)
test = test.drop(["employee_id", "region"], axis=1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
a = train.head()
# consumer and provider split
provider8 = train.loc[train["department_Sales & Marketing"] == 1]# 16840
provider7 = train.loc[train["department_Operations"] == 1]#11348
provider6 = train.loc[train["department_Procurement"] == 1]#7138
provider5 = train.loc[train["department_Technology"] == 1]#7138
provider4 = train.loc[train["department_Analytics"] == 1]#5352
provider3 = train.loc[train["department_Finance"] == 1]#2536
provider2 = train.loc[train["department_HR"] == 1]#2418
provider1 = train.loc[train["department_Legal"] == 1]#1039
consumer = train.loc[train["department_R&D"] == 1]#999

consumer = consumer.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider1 = provider1.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider2 = provider2.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider3 = provider3.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider4 = provider4.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider5 = provider5.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider6 = provider6.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider7 = provider7.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
provider8 = provider8.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)

# one hot encoding for the train set
# consumer = pd.get_dummies(consumer)
# provider1 = pd.get_dummies(provider1)
# provider2 = pd.get_dummies(provider2)
# provider3 = pd.get_dummies(provider3)
# provider4 = pd.get_dummies(provider4)
# provider5 = pd.get_dummies(provider5)
# provider6 = pd.get_dummies(provider6)
# provider7 = pd.get_dummies(provider7)
# provider8 = pd.get_dummies(provider8)

np.save("data/classification2/consumer.npy", consumer)
np.save("data/classification2/provider1.npy", provider1)
np.save("data/classification2/provider2.npy", provider2)
np.save("data/classification2/provider3.npy", provider3)
np.save("data/classification2/provider4.npy", provider4)
np.save("data/classification2/provider5.npy", provider5)
np.save("data/classification2/provider6.npy", provider6)
np.save("data/classification2/provider7.npy", provider7)
np.save("data/classification2/provider8.npy", provider8)


test = test.drop(
    [
        "department_Sales & Marketing",
        "department_Operations",
        "department_Procurement",
        "department_Technology",
        "department_Analytics",
        "department_Finance",
        "department_HR",
        "department_Legal",
        "department_R&D",
    ],
    axis=1,
)
test = pd.get_dummies(test)

np.save("data/classification2/test.npy", test)

# splitting the train set into dependent and independent sets
x_sample = train.iloc[:, :-1]
y_sample = train.iloc[:, -1]
# one hot encoding for the train set
x_sample = pd.get_dummies(x_sample)

x_sample = pd.DataFrame(x_sample)
y_sample = pd.DataFrame(y_sample)


from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(
    x_sample, y_sample, test_size=0.2, random_state=0
)

# standard scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
y_train = y_train.values
y_valid = y_valid.values
# # saving the employee_id
# emp_id = test["employee_id"]
# # removing the employee_id column
# test = test.drop(["employee_id"], axis=1)
# # defining the test set
# x_test = test
# print("Size of x-sample :", x_test.shape)
# # one hot encoding for the test set
# x_test = pd.get_dummies(x_test)

# checking the sizes of the sample data
# print("Size of x-sample :", x_sample.shape)
# print("Size of y-sample :", y_sample.shape)
np.save("data/classification2/x_train.npy", x_train)
np.save("data/classification2/x_valid.npy", x_valid)
np.save("data/classification2/y_train.npy", y_train)
np.save("data/classification2/y_valid.npy", y_valid)
print()
