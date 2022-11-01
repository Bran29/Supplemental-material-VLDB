import pandas as pd
from Animal90 import Animal
import torchvision
import torch
from tqdm import tqdm
import sys
import os.path as osp
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from itertools import permutations
import random
import copy
import HouseModels
from sklearn.model_selection import train_test_split

device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


main_dir = osp.dirname(osp.dirname(__file__))

# lib_path = osp.join(this_dir, "..", "OTDD")
add_path(main_dir)

from OTDD.otdd_pytorch import *

#  otdd distance
distance_tensorized = PytorchEuclideanDistance()
routine_tensorized = SinkhornTensorized(distance_tensorized)
cost_tensorized = SamplesLossTensorized(routine_tensorized)

predicate_set_animal = [
    (["Animal6", "label", 1], 0.36),
    (["Animal3", "label", 1], 0.29),
    (["Animal151", "label", 1], 0.35),
    (["Animal151", "label", 2], 0.35),
    (["Animal151", "label", 3], 0.35),
    (["Animal151", "label", 4], 0.35),
    (["Animal151", "label", 5], 0.35),
    (["cifar10", "label", 1], 0.31),
    (["cifar10", "label", 2], 0.31),
    (["cifar10", "label", 5], 0.31),
    (["cifar100", "label", 1], 0.32),
    (["cifar100", "label", 3], 0.32),
    (["cifar100", "label", 4], 0.32),
]

predicate_set_house = [
    (["provider1", "Price", 20, 100], 0.11),
    (["provider1", "Price", 100, 500], 0.11),
    (["provider1", "Price", 500, 1000], 0.11),
    (["provider1", "Price", 1000, 4000], 0.11),
    (["provider2", "Price", 20, 100], 0.09),
    (["provider2", "Price", 100, 500], 0.09),
    (["provider2", "Price", 500, 1000], 0.09),
    (["provider2", "Price", 1000, 4000], 0.09),
    (["provider3", "Price", 20, 100], 0.12),
    (["provider3", "Price", 100, 500], 0.12),
    (["provider3", "Price", 500, 1000], 0.12),
    (["provider3", "Price", 1000, 4000], 0.12),
    (["provider4", "Price", 20, 100], 0.10),
    (["provider4", "Price", 100, 500], 0.10),
    (["provider4", "Price", 500, 1000], 0.10),
    (["provider4", "Price", 1000, 4000], 0.10),
    (["provider5", "Price", 20, 100], 0.08),
    (["provider5", "Price", 100, 500], 0.08),
    (["provider5", "Price", 500, 1000], 0.08),
    (["provider5", "Price", 1000, 4000], 0.08),
]
predicate_set_hr = [
    (["provider1", "gender_m", 1], 0.11),
    (["provider1", "gender_m", 0], 0.11),
    (["provider1", "age", 20, 40], 0.11),
    (["provider1", "age", 40, 60], 0.11),
    (["provider2", "gender_m", 1], 0.12),
    (["provider2", "gender_m", 0], 0.12),
    (["provider2", "age", 20, 40], 0.12),
    (["provider2", "age", 40, 60], 0.12),
    (["provider3", "gender_m", 1], 0.11),
    (["provider3", "gender_m", 0], 0.11),
    (["provider3", "age", 20, 40], 0.11),
    (["provider3", "age", 40, 60], 0.11),
    (["provider4", "gender_m", 1], 0.09),
    (["provider4", "gender_m", 0], 0.09),
    (["provider4", "age", 20, 40], 0.09),
    (["provider4", "age", 40, 60], 0.09),
    (["provider5", "gender_m", 1], 0.10),
    (["provider5", "gender_m", 0], 0.10),
    (["provider5", "age", 20, 40], 0.10),
    (["provider5", "age", 40, 60], 0.10),
    (["provider6", "gender_m", 1], 0.08),
    (["provider6", "gender_m", 0], 0.08),
    (["provider6", "age", 20, 40], 0.08),
    (["provider6", "age", 40, 60], 0.08),
    (["provider7", "gender_m", 1], 0.12),
    (["provider7", "gender_m", 0], 0.12),
    (["provider7", "age", 20, 40], 0.12),
    (["provider7", "age", 40, 60], 0.12),
    (["provider8", "gender_m", 1], 0.1),
    (["provider8", "gender_m", 0], 0.1),
    (["provider8", "age", 20, 40], 0.1),
]


def tensor_2_dataframe(dataset):
    df = pd.DataFrame(dataset)
    names = []
    for i in range(dataset.shape[1] - 1):
        names.append("feature" + str(i))
    names.append("label")
    df.columns = names
    return df


def prepare_dataset():
    animal_owned = None
    employee_owned = None
    house_owned = None

    animal = dict()
    employee = dict()
    house = dict()

    # animal
    # transform = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.Resize([32, 32]),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),
    #     ]
    # )
    # datasets_name = [
    #     # "Animal6",
    #     # "Animal3",
    #     "Animal151",
    #     # "cifar10",
    #     # "cifar100",
    # ]

    # # consumer dataset load
    # dataset = Animal(
    #     root="data/classfication1/Animal90",
    #     train_rate=1,
    #     train=True,
    #     transform=transform,
    # )
    # dataset = torch.utils.data.DataLoader(
    #     dataset, batch_size=len(dataset), shuffle=True, num_workers=2
    # )
    # data_iter = iter(dataset)
    # dataset_x, dataset_y = data_iter.next()
    # dataset_x = dataset_x[:, 0].view(-1, 1024)
    # dataset_y = dataset_y.view(-1, 1)
    # dataset_all = torch.hstack((dataset_x, dataset_y))
    # df = pd.DataFrame(dataset_all)
    # names = []
    # for i in range(dataset_all.shape[1] - 1):
    #     names.append("feature_" + str(i))
    # names.append("label")
    # df.columns = names
    # animal_owned = df

    # # provider load
    # for file_name in tqdm(datasets_name):
    #     dataset = Animal(
    #         root="data/classfication1/" + file_name,
    #         train_rate=1,
    #         train=True,
    #         transform=transform,
    #     )
    #     dataset = torch.utils.data.DataLoader(
    #         dataset, batch_size=len(dataset), shuffle=True, num_workers=2
    #     )
    #     data_iter = iter(dataset)
    #     dataset_x, dataset_y = data_iter.next()
    #     dataset_x = dataset_x[:, 0].view(-1, 1024)
    #     dataset_y = dataset_y.view(-1, 1)
    #     dataset_all = torch.hstack((dataset_x, dataset_y))
    #     df = pd.DataFrame(dataset_all)
    #     names = []
    #     for i in range(dataset_all.shape[1] - 1):
    #         names.append("feature_" + str(i))
    #     names.append("label")
    #     df.columns = names
    #     animal[file_name] = df

    # house
    base_folder = "data/regression1/"
    datasets_name = [
        "provider1",
        "provider2",
        "provider3",
        "provider4",
        "provider5",
    ]

    file_path = base_folder + "consumer.npy"
    df_dataset = np.load(file_path, allow_pickle=True)
    # find name
    df = pd.read_csv("data/regression1/Indian House Prices.csv", index_col=False)
    df = df.iloc[:, 1:]

    df = df.drop(["Location"], axis=1)
    df = pd.get_dummies(df)

    consumer = df.loc[df["City_Mumbai"] == 1]  # 1302

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

    name = consumer.columns

    # sc = StandardScaler()
    # dataset = sc.fit_transform(df_dataset)
    # df_dataset = pd.DataFrame(dataset)
    # df_dataset.columns = name

    house_owned = consumer

    for file_name in tqdm(datasets_name):
        file_path = base_folder + file_name + ".npy"
        df_dataset = pd.DataFrame(np.load(file_path, allow_pickle=True))
        # dataset = sc.transform(df_dataset)
        # df_dataset = pd.DataFrame(dataset)
        df_dataset.columns = name

        house[file_name] = df_dataset

    # # employee
    # base_folder = "data/classification2/"
    # datasets_name = [
    #     "provider1",
    #     "provider2",
    #     "provider3",
    #     "provider4",
    #     "provider5",
    #     "provider6",
    #     "provider7",
    #     "provider8",
    # ]

    # file_path = base_folder + "consumer.npy"
    # df_dataset = np.load(file_path, allow_pickle=True)
    # # find name
    # train = pd.read_csv("data/classification2/train.csv")
    # train["education"].fillna(train["education"].mode()[0], inplace=True)
    # train["previous_year_rating"].fillna(1, inplace=True)
    # train = train.drop(["employee_id", "region"], axis=1)
    # train = pd.get_dummies(train)

    # consumer = train.loc[train["department_R&D"] == 1]  # 999
    # consumer = consumer.drop(
    #     [
    #         "department_Sales & Marketing",
    #         "department_Operations",
    #         "department_Procurement",
    #         "department_Technology",
    #         "department_Analytics",
    #         "department_Finance",
    #         "department_HR",
    #         "department_Legal",
    #         "department_R&D",
    #     ],
    #     axis=1,
    # )
    # name = consumer.columns

    # sc = StandardScaler()
    # dataset = sc.fit_transform(df_dataset)
    # df_dataset = pd.DataFrame(dataset)
    # df_dataset.columns = name

    # consumer_owned = df_dataset

    # for file_name in tqdm(datasets_name):
    #     file_path = base_folder + file_name + ".npy"
    #     df_dataset = pd.DataFrame(np.load(file_path, allow_pickle=True))
    #     dataset = sc.transform(df_dataset)
    #     df_dataset = pd.DataFrame(dataset)
    #     df_dataset.columns = name

    #     employee[file_name] = df_dataset

    return animal_owned, animal, employee_owned, employee, house_owned, house


def random_sampling(dataset, predicate, sample_size):
    if len(predicate[0]) == 3:
        Rq = dataset.loc[dataset[predicate[0][1]] == predicate[0][2]].sample(
            sample_size
        )

        dataset.drop(index=Rq.index, inplace=True)
        Rq.reset_index(drop=True, inplace=True)
        dataset.reset_index(drop=True, inplace=True)
    if len(predicate[0]) == 4:
        Rq = dataset.loc[
            (dataset[predicate[0][1]] >= predicate[0][2])
            & (dataset[predicate[0][1]] < predicate[0][3])
        ].sample(sample_size)
        dataset.drop(index=Rq.index, inplace=True)
        Rq.reset_index(drop=True, inplace=True)
        dataset.reset_index(drop=True, inplace=True)

    return Rq
    # labels = df_dataset["label"].value_counts()
    # print()
    # stratified_x = [
    #     torch.stack(
    #         [
    #             df_dataset.iloc[j]
    #             for j in range(df_dataset.shape[0])
    #             if df_dataset["label"].iloc[j] == i
    #         ],
    #         0,
    #     )
    #     for i in labels[0]
    # ]
    # print("end")


def exploration(datasets=None, B=100, dB=10, predicate_list=[]):

    # predicate:[['feature',6,0],['label',1]]
    # initial queryied amount
    I = [0] * len(predicate_list)
    # queried_set
    queried_set = []
    # current step
    tt = 0
    # dataset
    RP_set = []
    # tentative query
    for i, predicate in enumerate(predicate_list):
        # queried dataset
        dI = int(dB / predicate[1])
        dataset = datasets[predicate[0][0]]
        if len(predicate[0]) == 3:
            aim_set = dataset.loc[dataset[predicate[0][1]] == predicate[0][2]]
        else:
            aim_set = dataset.loc[
                (dataset[predicate[0][1]] >= predicate[0][2])
                & (dataset[predicate[0][1]] < predicate[0][3])
            ]
        if dI > aim_set.shape[0]:
            dI = aim_set.shape[0]
        Rq = random_sampling(dataset, predicate, dI)
        # update consumer set
        queried_set.append(Rq)
        # Dc = pd.concat([Dc, Rq])
        # update remain budget
        B -= dB
        # queried amount
        I[i] += dI
        # budget --->0 return
        if B <= 0:
            return queried_set, B
    # greedy query
    while B >= dB:
        rewards = []
        for i, predicate in enumerate(predicate_list):
            dI = int(dB / predicate[1])
            dataset = datasets[predicate[0][0]]
            if len(predicate[0]) == 3:
                aim_set = dataset.loc[dataset[predicate[0][1]] == predicate[0][2]]
            else:
                aim_set = dataset.loc[
                    (dataset[predicate[0][1]] >= predicate[0][2])
                    & (dataset[predicate[0][1]] < predicate[0][3])
                ]
            if dI > aim_set.shape[0]:
                reward = -1
            else:
                reward = (math.sqrt(dI / I[i] + 1) - 1) / (
                    predicate[1] * dI * math.sqrt(dI / I[i] + 1)
                )
            rewards.append(reward)

        max_index = rewards.index(max(rewards))
        # dataset has been explored entirely
        if rewards[max_index] < 0:
            return queried_set, B

        # queried dataset
        dataset = datasets[predicate_list[max_index][0][0]]
        Rq = random_sampling(
            dataset, predicate_list[max_index], int(dB / predicate_list[max_index][1])
        )
        # update consumer set
        queried_set[max_index] = pd.concat([queried_set[max_index], Rq])
        # update remain budget
        B -= dB
        # queried amount
        I[max_index] += dI

    return queried_set, B
    # for predicate in predicate_list:
    # for i,predicate in enumerate(predicate_list):
    #     dI=int(dB/price_list[i])
    #     Dc.append()


def calculate_single_utility(queried_data, consumer_data):
    # delete label and is_promoted
    if "label" in queried_data.columns:
        queried_data = queried_data.drop(columns=["label"])
        consumer_data = consumer_data.drop(columns=["label"])
    if "is_promoted" in queried_data.columns:
        queried_data = queried_data.drop(columns=["is_promoted"])
        consumer_data = consumer_data.drop(columns=["is_promoted"])

    max_num = queried_data.shape[0]
    if queried_data.shape[0] > consumer_data.shape[0]:
        max_num = consumer_data.shape[0]
    queried_data = torch.tensor(queried_data.sample(max_num).values.astype(np.float64))
    consumer_data = torch.tensor(
        consumer_data.sample(max_num).values.astype(np.float64)
    )

    distance = cost_tensorized.distance(queried_data, consumer_data)[0]

    return distance


def calculate_utility(queried_set, Dc):
    utility_set = []
    for queried_data in queried_set:
        utility_set.append(calculate_single_utility(queried_data, Dc))
    return utility_set


def SV(queried_set, Dc):
    M = len(queried_set)
    orderings = []
    if M > 4:
        for i in range(M):
            M_list = random.sample(range(M), 4)
            orderings += list(permutations(M_list))
    else:
        orderings = list(permutations(range(M)))

    monte_carlo_s_values = torch.zeros(M)
    # Monte-carlo : shuffling the ordering and taking the first K orderings
    random.shuffle(orderings)
    K = 100  # number of permutations to sample

    for ordering_count, ordering in tqdm(enumerate(orderings)):
        prefix_value = 0
        for position, i in enumerate(ordering):
            curr_indices = set(ordering[: position + 1])
            curr_set = pd.concat([queried_set[j] for j in curr_indices])
            curr_value = calculate_single_utility(curr_set, Dc)

            marginal = curr_value - prefix_value
            prefix_value = curr_value

            monte_carlo_s_values[i] += marginal
        if ordering_count > K:
            break
        print(ordering_count, "/", len(orderings))
    monte_carlo_s_values /= K
    return monte_carlo_s_values


def CombinationSearch(method="SV"):
    return


def UCB(queried_set=[], Dc=None, utility_set=[]):
    N = len(queried_set)
    # init
    k = 0
    K = 100
    M = N
    Rp_history = []

    reward = utility_set
    ucb_value = utility_set

    alpha = [0.1] * M
    # alpha=list(np.linspace(0.1,0.2,M))
    yita = 1
    n_k = [0] * N
    n_all = 0
    while k < min(K, math.factorial(N)):

        Rp_history = []

        reward_prior = 0

        current_ucb_value = [-1 if j in Rp_history else ucb_value[j] for j in range(N)]
        for i in range(M):
            maxucb_i = current_ucb_value.index(max(current_ucb_value))
            Rp_history.append(maxucb_i)
            curr_set = pd.concat([queried_set[j] for j in Rp_history])
            reward_after = calculate_single_utility(curr_set, Dc)
            dr = reward_after - reward_prior
            if dr > 0:
                n_all += 1
                n_k[maxucb_i] += 1
                for j in range(N):
                    if j in Rp_history:
                        reward[j] += alpha[j] * dr
                        ucb_value[j] = reward[j] + yita * math.sqrt(
                            2 * math.log(n_all / n_k[j])
                        )
                        # print(reward[j])
                        # print(alpha[j] * dr)
                        # print(yita * math.sqrt(2 * math.log(n_all / n_k[j])))
                        # print()
                    else:
                        ucb_value[j] = reward[j] + yita * math.sqrt(
                            2 * math.log(n_all / (n_k[j] + 0.001))
                        )
            else:
                del Rp_history[-1]

            current_ucb_value = [
                -1 if j in Rp_history else ucb_value[j] for j in range(N)
            ]

            reward_prior = reward_after
        k += 1

    return [value / K for value in reward]


def allocate_non_combination(
    datasets=None, budget=400, utility_set=[], predicate_list=[], queried_set=[]
):
    sorted_id = sorted(
        range(len(utility_set)), key=lambda k: utility_set[k], reverse=True
    )
    remaining_budget = budget
    for i in sorted_id:
        if remaining_budget <= 0:
            return queried_set
        dataset = datasets[predicate_list[i][0][0]]
        if len(predicate_list[i][0]) == 3:
            aim_set = dataset.loc[
                dataset[predicate_list[i][0][1]] == predicate_list[i][0][2]
            ]
        else:
            aim_set = dataset.loc[
                (dataset[predicate_list[i][0][1]] >= predicate_list[i][0][2])
                & (dataset[predicate_list[i][0][1]] < predicate_list[i][0][3])
            ]
        allocate_budget = (utility_set[i] / sum(utility_set)) * remaining_budget
        dI = int(allocate_budget / predicate_list[i][1])
        if aim_set.shape[0] != 0:
            if dI > aim_set.shape[0]:
                dI = aim_set.shape[0]
                allocate_budget = dI * predicate_list[i][1]
            Rq = random_sampling(dataset, predicate_list[i], dI)
            queried_set[i] = pd.concat([queried_set[i], Rq])
        else:
            allocate_budget = 0
        remaining_budget -= allocate_budget


def allocate(
    method="c",
    datasets=None,
    budget=400,
    score_set=[],
    predicate_list=[],
    queried_set=[],
):
    sorted_id = sorted(range(len(score_set)), key=lambda k: score_set[k], reverse=True)
    remaining_budget = budget
    if method == "c":
        for i in sorted_id:
            if remaining_budget <= 0:
                return queried_set
            dataset = datasets[predicate_list[i][0][0]]

            if len(predicate_list[i][0]) == 3:
                aim_set = dataset.loc[
                    dataset[predicate_list[i][0][1]] == predicate_list[i][0][2]
                ]
            else:
                aim_set = dataset.loc[
                    (dataset[predicate_list[i][0][1]] >= predicate_list[i][0][2])
                    & (dataset[predicate_list[i][0][1]] < predicate_list[i][0][3])
                ]
            allocate_budget = (score_set[i] / sum(score_set)) * remaining_budget
            dI = int(allocate_budget / predicate_list[i][1])
            if aim_set.shape[0] != 0:
                if dI > aim_set.shape[0]:
                    dI = aim_set.shape[0]
                    allocate_budget = dI * predicate_list[i][1]
                Rq = random_sampling(dataset, predicate_list[i], dI)
                queried_set[i] = pd.concat([queried_set[i], Rq])
            else:
                allocate_budget = 0
            remaining_budget -= allocate_budget

    if method == "c/p":
        for i in range(len(score_set)):
            if remaining_budget <= 0:
                return queried_set
            dataset = datasets[predicate_list[i][0][0]]
            aim_set = dataset.loc[
                dataset[predicate_list[i][0][1]] == predicate_list[i][0][2]
            ]
            allocate_budget = (
                (score_set[i] / predicate_list[i][1])
                / sum(
                    [score_set[j] / predicate_list[j][1] for j in range(len(score_set))]
                )
            ) * budget
            if aim_set.shape[0] != 0:
                if dI > aim_set.shape[0]:
                    dI = aim_set.shape[0]
                    allocate_budget = dI * predicate_list[i][1]
                Rq = random_sampling(dataset, predicate_list[i], dI)
                queried_set[i] = pd.concat([queried_set[i], Rq])
            else:
                allocate_budget = 0
            remaining_budget -= allocate_budget

    return queried_set


def House_run(queried_set, Dc, X_test, y_test):
    q_dataset = pd.concat(queried_set)

    q_dataset_x = q_dataset[q_dataset.columns.difference(["Price"])]
    q_dataset_y = q_dataset["Price"]

    X_train, _, y_train, _ = train_test_split(
        Dc[Dc.columns.difference(["Price"])],
        Dc["Price"],
        test_size=0.01,
        random_state=42,
    )
    X_train = pd.concat([X_train, q_dataset_x]).values

    y_train = pd.concat([y_train, q_dataset_y]).values

    X_test = X_test.values
    y_test = y_test.values

    regression_model = HouseModels.regression(X_train, y_train, X_test, y_test)
    mse = regression_model.adaboost()
    return mse


if __name__ == "__main__":
    (
        animal_owned_real,
        animal_real,
        employee_owned_real,
        employee_real,
        house_owned_real,
        house_real,
    ) = prepare_dataset()

    all_house = pd.concat(
        [house_owned_real, pd.concat([data for data in house_real.values()])]
    )

    _, X_test, _, y_test = train_test_split(
        all_house[all_house.columns.difference(["Price"])],
        all_house["Price"],
        test_size=0.01,
        random_state=42,
    )

    # main
    # all_budget_list = [100, 1000, 2000, 5000, 10000]
    mse_list = []
    # explore_rate = 0.2
    explore_rate_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    all_budget = 2000
    # for all_budget in all_budget_list:
    for explore_rate in explore_rate_list:
        house_owned = copy.deepcopy(house_owned_real)
        house = copy.deepcopy(house_real)

        queried_set, remain_budget = exploration(
            datasets=house,
            B=explore_rate * all_budget,
            predicate_list=predicate_set_house,
        )
        utility_set = calculate_utility(queried_set, house_owned)
        # score1 = SV(queried_set, house_owned)
        score2 = UCB(queried_set, house_owned, utility_set)
        queried_set = allocate(
            method="c",
            datasets=house,
            budget=all_budget * (1 - explore_rate) + remain_budget,
            score_set=score2,
            predicate_list=predicate_set_house,
            queried_set=queried_set,
        )
        mse = House_run(queried_set, house_owned, X_test, y_test)
        mse_list.append(mse)
        np.save("house_list_6.4.npy", mse_list)
    print(mse_list)
    np.save("house_list_6.4.npy", mse_list)
    # transform = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.Resize([32, 32]),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),
    #     ]
    # )
    # # load cifar
    # dataset = Animal(
    #     root="data/classfication1/cifar10",
    #     train_rate=1,
    #     train=True,
    #     transform=transform,
    # )
    # dataset = torch.utils.data.DataLoader(
    #     dataset, batch_size=len(dataset), shuffle=True, num_workers=2
    # )
    # data_iter = iter(dataset)
    # dataset_x, dataset_y = data_iter.next()
    # dataset_x = dataset_x[:, 0].view(-1, 1024)
    # dataset_y = dataset_y.view(-1, 1)
    # dataset_all = torch.hstack((dataset_x, dataset_y))
    # tensor_2_dataframe(dataset_all)
