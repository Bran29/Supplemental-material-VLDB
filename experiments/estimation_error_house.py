from random import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import sys
import os.path as osp
from sklearn.cluster import Birch, KMeans
import random


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


main_dir = osp.dirname(osp.dirname(__file__))

# lib_path = osp.join(this_dir, "..", "OTDD")
add_path(main_dir)

from OTDD.otdd_pytorch import *

# Birch + add label
def clustering(X, n_cluster=3):
    # y_pre = Birch(n_clusters=n_cluster).fit_predict(X)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(X)
    y_pre = kmeans.labels_
    return y_pre


def random_sampling(dataset_x, sampling_rate=None, N=None, k=0, n_cluster=3):
    sampling_times = len(sampling_rate)
    np.random.seed(k)
    np.random.shuffle(dataset_x)
    dataset_x = torch.tensor(dataset_x)
    sampled_dataset = []
    for t in range(sampling_times):
        if t == 0:
            sampled_dataset.append(dataset_x[: int(N * sampling_rate[t])])
        else:
            sampled_dataset.append(
                dataset_x[int(N * sampling_rate[t - 1]) : int(N * sampling_rate[t])]
            )
    return sampled_dataset


def stratified_sampling(dataset_x, sampling_rate=None, N=None, k=0, n_cluster=3):
    sampling_times = len(sampling_rate)
    sampled_dataset = []

    dataset_y = clustering(dataset_x, n_cluster=n_cluster).reshape(
        (dataset_x.shape[0], 1)
    )
    dataset_new = np.hstack((dataset_x, dataset_y))

    np.random.seed(k)
    np.random.shuffle(dataset_new)

    dataset_x = dataset_new[:, :-1]
    dataset_y = dataset_new[:, -1]

    dataset_x = torch.tensor(dataset_x)
    dataset_y = torch.tensor(dataset_y)

    labels = torch.unique(dataset_y, return_counts=True)
    stratified_n = [
        N * labels[1] / sum(labels[1]) * sampling_rate[t] for t in range(sampling_times)
    ]
    stratified_n = [stratified_n[t].int() for t in range(sampling_times)]

    for t in range(sampling_times):
        if sum(stratified_n[t]) != int(N * sampling_rate[t]):
            i = labels[1].argmax()
            stratified_n[t][i] += int(N * sampling_rate[t]) - sum(stratified_n[t])

    stratified_y = []
    for t in range(sampling_times):
        stratified_y_list = []
        for i in range(labels[0].shape[0]):
            if t == 0:
                for j in range(stratified_n[t][i]):
                    stratified_y_list.append(labels[0][i])
            else:
                for j in range(stratified_n[t][i] - stratified_n[t - 1][i]):
                    stratified_y_list.append(labels[0][i])
        stratified_y_list = torch.tensor(stratified_y_list)
        stratified_y.append(stratified_y_list)

    stratified_x = [
        torch.stack(
            [dataset_x[j] for j in range(dataset_y.shape[0]) if dataset_y[j] == i], 0
        )
        for i in labels[0]
    ]
    for t in range(sampling_times):
        if t == 0:
            stratified_sample_x = torch.cat(
                [
                    stratified_x[i][: stratified_n[t][i]]
                    for i in range(labels[0].shape[0])
                ],
                dim=0,
            )
        else:
            stratified_sample_x = torch.cat(
                [
                    stratified_x[i][stratified_n[t - 1][i] : stratified_n[t][i]]
                    for i in range(labels[0].shape[0])
                ],
                dim=0,
            )

        sampled_dataset.append(stratified_sample_x)
    return sampled_dataset


def stratified_nayman_sampling(dataset_x, sampling_rate=0.1, N=None, k=0, n_cluster=3):
    sampling_times = len(sampling_rate)
    sampled_dataset = []

    dataset_y = clustering(dataset_x, n_cluster=n_cluster).reshape(
        (dataset_x.shape[0], 1)
    )
    dataset_new = np.hstack((dataset_x, dataset_y))

    np.random.seed(k)
    np.random.shuffle(dataset_new)

    dataset_x = dataset_new[:, :-1]
    dataset_y = dataset_new[:, -1]

    dataset_x = torch.tensor(dataset_x)
    dataset_y = torch.tensor(dataset_y)

    labels = torch.unique(dataset_y, return_counts=True)

    stratified_x = [
        torch.stack(
            [dataset_x[j] for j in range(dataset_y.shape[0]) if dataset_y[j] == i],
            dim=0,
        )
        for i in labels[0]
    ]

    # calculate sigma
    sigma = torch.stack(
        [
            torch.mean(
                torch.sqrt(
                    torch.mean(
                        torch.abs(stratified_x[i] - torch.mean(stratified_x[i], dim=0))
                        ** 2,
                        dim=0,
                    )
                )
            )
            for i in range(labels[0].shape[0])
        ],
        0,
    )
    w = torch.stack([n / N for n in labels[1]], 0)
    b = w.mul(sigma)
    # print(b)
    stratified_n = [
        [
            ((w[i] * sigma[i]) / torch.sum(w.mul(sigma)) * (N * sampling_rate[t]))
            .ceil()
            .int()
            for i in range(labels[1].shape[0])
        ]
        for t in range(sampling_times)
    ]
    for t in range(sampling_times):
        if sum(stratified_n[t]) != int(N * sampling_rate[t]):
            i = labels[1].argmax()
            stratified_n[t][i] += int(N * sampling_rate[t]) - sum(stratified_n[t])

    stratified_y = []
    for t in range(sampling_times):
        stratified_y_list = []
        for i in range(labels[0].shape[0]):
            if t == 0:
                for j in range(stratified_n[t][i]):
                    stratified_y_list.append(labels[0][i])
            else:
                for j in range(stratified_n[t][i] - stratified_n[t - 1][i]):
                    stratified_y_list.append(labels[0][i])
        stratified_y_list = torch.tensor(stratified_y_list)
        stratified_y.append(stratified_y_list)

    for t in range(sampling_times):
        if t == 0:
            stratified_sample_x = torch.cat(
                [
                    stratified_x[i][: stratified_n[t][i]]
                    for i in range(labels[0].shape[0])
                ],
                dim=0,
            )
        else:
            stratified_sample_x = torch.cat(
                [
                    stratified_x[i][stratified_n[t - 1][i] : stratified_n[t][i]]
                    for i in range(labels[0].shape[0])
                ],
                dim=0,
            )
        sampled_dataset.append(stratified_sample_x)
    return sampled_dataset


def to_do(dataset1, dataset2, sample_method, n_cluster=3):
    distance_tensorized = PytorchEuclideanDistance()
    routine_tensorized = SinkhornTensorized(distance_tensorized)
    cost_tensorized = SamplesLossTensorized(routine_tensorized)

    error = []
    for k in range(20):
        sampling_rate = list(np.linspace(0.01, 0.09, 9)) + list(
            np.linspace(0.1, 1.0, 10)
        )

        N = 2000
        dataset1_list = sample_method(
            dataset1, sampling_rate=sampling_rate, N=N, k=k, n_cluster=n_cluster
        )
        dataset2_list = sample_method(
            dataset2, sampling_rate=sampling_rate, N=N, k=k, n_cluster=n_cluster
        )

        dataset1_x_cat = []
        dataset2_x_cat = []

        for i in range(len(dataset1_list)):
            dataset1_x_cat.append(
                torch.cat([dataset1_list[j] for j in range(i + 1)], 0)
            )

            dataset2_x_cat.append(
                torch.cat([dataset2_list[j] for j in range(i + 1)], 0)
            )

            print(dataset1_list[i].shape)
            print(dataset2_list[i].shape)

        distance_all = cost_tensorized.distance(dataset1_x_cat[-1], dataset2_x_cat[-1])[
            0
        ]

        error_list = []
        for s in range(len(sampling_rate)):
            distance_tmp = cost_tensorized.distance(
                dataset1_x_cat[s], dataset2_x_cat[s]
            )[0]
            error_list.append((abs(distance_tmp - distance_all) / distance_all).numpy())
        error.append(error_list)
    error = np.array(error)
    np.save("result/house_" + str(sample_method) + ".npy", error)


if __name__ == "__main__":
    consumer = pd.DataFrame(
        np.load("data/regression1/provider4.npy", allow_pickle=True)
    )
    provider5 = pd.DataFrame(
        np.load("data/regression1/provider5.npy", allow_pickle=True)
    )
    sc = StandardScaler()

    # normalized
    sc = StandardScaler()
    consumer = sc.fit_transform(consumer)
    provider5 = sc.transform(provider5)

    # to_do(consumer, provider5, random_sampling, n_cluster=3)
    to_do(consumer, provider5, stratified_sampling, n_cluster=5)
    # to_do(consumer, provider5, stratified_nayman_sampling, n_cluster=3)
