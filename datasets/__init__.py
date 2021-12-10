# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)
