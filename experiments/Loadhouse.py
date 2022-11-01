import pandas as pd
import sys
import os.path as osp
import numpy as np

# For data preprocessing
from geopy.geocoders import Nominatim


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


main_dir = osp.dirname(osp.dirname(__file__))

# lib_path = osp.join(this_dir, "..", "OTDD")
add_path(main_dir)

file_path = "data/regression1/Indian House Prices.csv"
df = pd.read_csv(file_path, index_col=False)
df = df.iloc[:, 1:]

print(df.columns)
df = df.drop(["Location"], axis=1)
df = pd.get_dummies(df)


consumer = df.loc[df["City_Mumbai"] == 1]  # 1302
provider1 = df.loc[df["City_Kolkata"] == 1]  # 68
provider2 = df.loc[df["City_Banglore"] == 1]  # 1708
provider4 = df.loc[df["City_Chennai"] == 1]  # 2047
provider3 = df.loc[df["City_Delhi"] == 1]  # 1989
provider5 = df.loc[df["City_Hyderabad"] == 1]  # 2276

consumer = consumer.drop(
    [
        "City_Banglore",
        "City_Banglore",
        "City_Delhi",
        "City_Hyderabad",
        "City_Kolkata",
        "City_Mumbai",
    ],
    axis=1,
)
provider1 = provider1.drop(
    [
        "City_Banglore",
        "City_Banglore",
        "City_Delhi",
        "City_Hyderabad",
        "City_Kolkata",
        "City_Mumbai",
    ],
    axis=1,
)
provider2 = provider2.drop(
    [
        "City_Banglore",
        "City_Banglore",
        "City_Delhi",
        "City_Hyderabad",
        "City_Kolkata",
        "City_Mumbai",
    ],
    axis=1,
)
provider3 = provider3.drop(
    [
        "City_Banglore",
        "City_Banglore",
        "City_Delhi",
        "City_Hyderabad",
        "City_Kolkata",
        "City_Mumbai",
    ],
    axis=1,
)
provider4 = provider4.drop(
    [
        "City_Banglore",
        "City_Banglore",
        "City_Delhi",
        "City_Hyderabad",
        "City_Kolkata",
        "City_Mumbai",
    ],
    axis=1,
)
provider5 = provider5.drop(
    [
        "City_Banglore",
        "City_Banglore",
        "City_Delhi",
        "City_Hyderabad",
        "City_Kolkata",
        "City_Mumbai",
    ],
    axis=1,
)
np.save("data/regression1/consumer.npy", consumer)
np.save("data/regression1/provider1.npy", provider1)
np.save("data/regression1/provider2.npy", provider2)
np.save("data/regression1/provider3.npy", provider3)
np.save("data/regression1/provider4.npy", provider4)
np.save("data/regression1/provider5.npy", provider5)
