import os.path
import pickle
from traceback import print_tb
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
import random


class Animal(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    file_name = "entry"

    def __init__(
        self,
        root: str,
        train: bool = True,
        train_rate: float = 0.8,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        file_path = os.path.join(self.root, self.file_name)
        with open(file_path, "rb") as f:
            entry = pickle.load(f)
            if self.train:
                self.data = entry["data"][: int(len(entry["data"]) * train_rate)]
                self.targets = entry["labels"][: int(len(entry["labels"]) * train_rate)]
            else:
                self.data = entry["data"][int(len(entry["data"]) * train_rate) :]
                self.targets = entry["labels"][int(len(entry["labels"]) * train_rate) :]

        self.data = np.vstack(self.data)
        shape_record = self.data.shape
        # print(shape_record)
        self.data = self.data.reshape(-1, shape_record[1], shape_record[1], 3)

    def _load_meta(self) -> None:
        self.classes = [
            "MammaliaDataset",
            "BirdDataset",
            "ReptilesDataset",
            "FishDataset",
            "AmphibianDataset",
        ]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize([32, 32]),
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_data = Animal(root="data", train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=True, num_workers=2
    )
    print("end. \n")
