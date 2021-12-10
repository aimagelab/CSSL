# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' online EWC.')
    add_management_args(parser)
    add_experiment_args(parser)
    parser.add_argument('--e_lambda', type=float, required=True,
                        help='lambda weight for EWC')
    parser.add_argument('--gamma', type=float, required=True,
                        help='gamma parameter for EWC online')

    return parser


class EwcOn(ContinualModel):
    NAME = 'ewc_on'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(EwcOn, self).__init__(backbone, loss, args, transform)

        self.logsoft = nn.LogSoftmax(dim=1)
        self.checkpoint = None
        self.Fish = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        else:
            penalty = (self.Fish * ((self.net.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset):
        Fish = torch.zeros_like(self.net.get_params())

        for j, data in enumerate(dataset.train_loader):
            inputs, labels, _ = data

            inputs = inputs[labels < 1000]
            labels = labels[labels < 1000]
            if not len(inputs):
                continue

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            for ex, lab in zip(inputs, labels):
                self.opt.zero_grad()
                output = self.net(ex.unsqueeze(0))
                loss = - F.nll_loss(self.logsoft(output), lab.unsqueeze(0),
                                    reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                Fish += exp_cond_prob * self.net.get_grads() ** 2
                # loss = self.loss(output, lab.unsqueeze(0))
                # loss.backward()
                # Fish += self.net.get_grads() ** 2

        Fish /= (len(dataset.train_loader) * self.args.batch_size)

        if self.Fish is None:
            self.Fish = Fish
        else:
            self.Fish *= self.args.gamma
            self.Fish += Fish

        self.checkpoint = self.net.get_params().data.clone()

    def observe(self, inputs, labels, not_aug_inputs):
        inputs, labels, not_aug_inputs, _ = self.drop_unlabeled(inputs, labels, not_aug_inputs)
        if len(inputs) == 0:
            return 0
            
        self.opt.zero_grad()
        outputs = self.net(inputs)
        penalty = self.penalty()
        loss = self.loss(outputs, labels) + self.args.e_lambda * penalty
        assert not torch.isnan(loss)
        loss.backward()
        self.opt.step()

        return loss.item()
