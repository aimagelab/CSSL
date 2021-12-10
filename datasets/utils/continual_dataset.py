# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import abstractmethod
from argparse import Namespace
from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torchvision import datasets
import numpy as np
import torch


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

    def prepare_data_loaders(self, train_dataset, test_dataset):
        if isinstance(train_dataset.targets, list) or not train_dataset.targets.dtype is torch.long:
            train_dataset.targets = torch.tensor(train_dataset.targets, dtype=torch.long)
        if isinstance(test_dataset.targets, list) or not test_dataset.targets.dtype is torch.long:
            test_dataset.targets = torch.tensor(test_dataset.targets, dtype=torch.long)

        if not hasattr(self, "label_mask"):
            setattr(self, 'label_mask', self.get_label_mask(train_dataset))

        if self.label_mask.sum() != 0:
            if hasattr(self.args, "semi_supervised_mode") and self.args.semi_supervised_mode == "pre-discard":
                train_dataset.data = np.delete(train_dataset.data, self.label_mask, axis=0)
                train_dataset.targets = np.delete(train_dataset.targets, self.label_mask, axis=0)
            else:
                train_dataset.targets[self.label_mask] += 1000

        return store_masked_loaders(train_dataset, test_dataset, self)

    def get_label_mask(self, train_dataset):
        """
        Creates a balanced mask for each class in the dataset.
        :param train_dataset: the entire training set
        :return: list: balanced masks for labels
        """
        ind = np.indices(train_dataset.targets.shape)[0]
        mask = []
        for i_label, _ in enumerate(train_dataset.classes):
            partial_targets = train_dataset.targets[train_dataset.targets == i_label]
            current_mask = np.random.choice(partial_targets.shape[0], max(
                partial_targets.shape[0] - self.args.lpc, 0), replace=False)

            mask = np.append(mask, ind[train_dataset.targets == i_label][current_mask])

        return mask.astype(np.int32)


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset, shuffletest: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    train_mask = np.logical_and(np.array(train_dataset.targets % 1000) >= setting.i,
        np.array(train_dataset.targets % 1000) < setting.i + setting.N_CLASSES_PER_TASK)
    test_mask = np.logical_and(np.array(test_dataset.targets % 1000) >= setting.i,
        np.array(test_dataset.targets % 1000) < setting.i + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=shuffletest, num_workers=0)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
