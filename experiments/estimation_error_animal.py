from random import random
import torch
import torchvision
import os.path as osp
import sys
import random


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


main_dir = osp.dirname(osp.dirname(__file__))

# lib_path = osp.join(this_dir, "..", "OTDD")
add_path(main_dir)

from OTDD.otdd_pytorch import *
import numpy as np
from Animal90 import Animal
import matplotlib.pyplot as plt


def random_sampling(dataset, sampling_rate=None, N=None):
    sampling_times = len(sampling_rate)
    sampled_dataset = []
    dataset = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=True, num_workers=2
    )
    data_iter = iter(dataset)
    dataset_x, dataset_y = data_iter.next()
    dataset_x = dataset_x[:, 0].view(-1, 1024)

    for t in range(sampling_times):
        if t == 0:
            sampled_dataset.append(
                [
                    dataset_x[: int(N * sampling_rate[t])],
                    dataset_y[: int(N * sampling_rate[t])],
                ]
            )
        else:
            sampled_dataset.append(
                [
                    dataset_x[
                        int(N * sampling_rate[t - 1]) : int(N * sampling_rate[t])
                    ],
                    dataset_y[
                        int(N * sampling_rate[t - 1]) : int(N * sampling_rate[t])
                    ],
                ]
            )
    return sampled_dataset


def stratified_sampling(dataset, sampling_rate=None, N=None):
    sampling_times = len(sampling_rate)
    sampled_dataset = []
    dataset = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=True, num_workers=2
    )
    data_iter = iter(dataset)
    dataset_x, dataset_y = data_iter.next()
    dataset_x = dataset_x[:, 0].view(-1, 1024)

    labels = torch.unique(dataset_y, return_counts=True)
    stratified_n = [
        N * labels[1] / sum(labels[1]) * sampling_rate[t] for t in range(sampling_times)
    ]
    stratified_n = [stratified_n[t].ceil().int() for t in range(sampling_times)]

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

    print(labels[0].shape)

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

        sampled_dataset.append([stratified_sample_x, stratified_y[t]])
    return sampled_dataset


def stratified_nayman_sampling(dataset, sampling_rate=None, N=None):
    sampling_times = len(sampling_rate)
    sampled_dataset = []
    dataset = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=True, num_workers=2
    )
    data_iter = iter(dataset)
    dataset_x, dataset_y = data_iter.next()
    dataset_x = dataset_x[:, 0].view(-1, 1024)
    labels = torch.unique(dataset_y, return_counts=True)

    stratified_x = [
        torch.stack(
            [dataset_x[j] for j in range(dataset_y.shape[0]) if dataset_y[j] == i],
            dim=0,
        )
        for i in labels[0]
    ]
    print(stratified_x[0].shape)
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
    stratified_n = [
        [
            ((w[i] * sigma[i]) / torch.sum(w.mul(sigma)) * (N * sampling_rate[t]))
            .ceil()
            .int()
            for i in range(labels[1].shape[0])
        ]
        for t in range(sampling_times)
    ]
    # stratified_n = [stratified_n[t].ceil() for t in range(sampling_times)]

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
        sampled_dataset.append([stratified_sample_x, stratified_y[t]])
    return sampled_dataset


# random sampling

# parameter settings
def to_do(dataset1, dataset2, sample_method):
    sampling_rate = list(np.linspace(0.01, 0.09, 9)) + list(np.linspace(0.1, 1.0, 10))

    #  otdd distance
    distance_tensorized = PytorchEuclideanDistance()
    routine_tensorized = SinkhornTensorized(distance_tensorized)
    cost_tensorized = SamplesLossTensorized(routine_tensorized)

    error = []
    for k in range(10):
        N = 2000

        dataset1_list = sample_method(dataset1, N=N, sampling_rate=sampling_rate)
        dataset2_list = sample_method(dataset2, N=N, sampling_rate=sampling_rate)

        dataset1_x_cat = []
        dataset2_x_cat = []
        dataset1_y_cat = []
        dataset2_y_cat = []
        for i in range(len(dataset1_list)):
            dataset1_x_cat.append(
                torch.cat([dataset1_list[j][0] for j in range(i + 1)], 0)
            )

            dataset2_x_cat.append(
                torch.cat([dataset2_list[j][0] for j in range(i + 1)], 0)
            )

            dataset1_y_cat.append(
                torch.cat([dataset1_list[j][1] for j in range(i + 1)], 0)
            )

            dataset2_y_cat.append(
                torch.cat([dataset2_list[j][1] for j in range(i + 1)], 0)
            )

            print(dataset1_list[i][0].shape)
            print(dataset2_list[i][0].shape)
            print(dataset1_list[i][1].shape)
            print(dataset2_list[i][1].shape)

        # distance_all = cost_tensorized.distance_with_labels(
        #     dataset1_x_cat[-1],
        #     dataset2_x_cat[-1],
        #     dataset1_y_cat[-1],
        #     dataset2_y_cat[-1],
        #     # gaussian_class_distance=True,
        # )[0]
        distance_all = cost_tensorized.distance(dataset1_x_cat[-1], dataset2_x_cat[-1])[
            0
        ]

        error_list = []
        for s in range(len(sampling_rate)):
            # distance_tmp = cost_tensorized.distance_with_labels(
            #     dataset1_x_cat[s],
            #     dataset2_x_cat[s],
            #     dataset1_y_cat[s],
            #     dataset2_y_cat[s],
            #     # gaussian_class_distance=True,
            # )[0]
            distance_tmp = cost_tensorized.distance(
                dataset1_x_cat[s], dataset2_x_cat[s]
            )[0]
            error_list.append(((distance_tmp - distance_all) / distance_all).numpy())
        error.append(error_list)
    error = np.array(error)
    # error_mean = np.nanmean(error, axis=0)
    np.save("result/animal_" + str(sample_method) + ".npy", error)
    # 4
    # N4 = min(consumer_images.shape[0], provider4_images.shape[0])
    # distance_all = cost_tensorized.distance_with_labels(
    #     consumer_images[:N4],
    #     provider4_images[:N4],
    #     consumer_labels[:N4],
    #     provider4_labels[:N4],
    # )[0]
    # error4_list = []
    # for s in sampling_rate:
    #     n = int(N4 * s)
    #     distance_tmp = cost_tensorized.distance_with_labels(
    #         consumer_images[:n],
    #         provider4_images[:n],
    #         consumer_labels[:n],
    #         provider4_labels[:n],
    #     )[0]
    #     error4_list.append((distance_tmp - distance_all) / distance_all)
    # estimate_error_animal4 = np.abs(np.array(error4_list))
    # np.save("result/estimate_error_animal4.npy", estimate_error_animal4)
    # 5
    # N5 = min(consumer_images.shape[0], provider5_images.shape[0])
    # distance_all = cost_tensorized.distance_with_labels(
    #     consumer_images[:N5],
    #     provider5_images[:N5],
    #     consumer_labels[:N5],
    #     provider5_labels[:N5],
    # )[0]
    # error5_list = []
    # for s in sampling_rate:
    #     n = int(N5 * s)
    #     distance_tmp = cost_tensorized.distance_with_labels(
    #         consumer_images[:n],
    #         provider5_images[:n],
    #         consumer_labels[:n],
    #         provider5_labels[:n],
    #     )[0]
    #     error5_list.append((distance_tmp - distance_all) / distance_all)
    # estimate_error_animal5 = np.abs(np.array(error5_list))
    # np.save("result/estimate_error_animal5.npy", estimate_error_animal5)


if __name__ == "__main__":
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    consumer_data = Animal(
        root="data/classfication1/Animal90",
        train_rate=1,
        train=True,
        transform=transform,
    )

    provider3_data = Animal(
        root="data/classfication1/Animal151",
        train_rate=1,
        train=True,
        transform=transform,
    )
    to_do(consumer_data, provider3_data, random_sampling)
    to_do(consumer_data, provider3_data, stratified_sampling)
    to_do(consumer_data, provider3_data, stratified_nayman_sampling)
    print()
