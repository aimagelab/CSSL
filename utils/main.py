# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from datasets import get_dataset
from models import get_model
from utils.training import train
import torch
from utils.conf import set_random_seed


def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--lpc', type=int, default=None,
                        help='Number of labeled examples per class.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)
    torch.set_num_threads(4)

    get_parser = getattr(mod, 'get_parser')
    parser = get_parser()
    parser.add_argument('--lpc', type=int, default=None,
                        help='Number of labeled examples per class.')
    args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    train(model, dataset, args)


if __name__ == '__main__':
    main()