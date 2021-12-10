# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from utils.args import *
from models.utils.continual_model import ContinualModel
from torch.optim import SGD
from utils.buffer import Buffer

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via'
                                        ' Progressive Neural Networks.')
    add_management_args(parser)
    add_rehearsal_args(parser)
    add_experiment_args(parser)
    return parser

def fit_buffer(self, opt_steps):
    for _ in range(opt_steps):
        self.opt.zero_grad()

        buf_inputs, buf_labels = self.buffer.get_data(
            self.args.minibatch_size, transform=self.transform)
        buf_outputs = self.net(buf_inputs)
        loss = self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.opt.step()

class GDumb(ContinualModel):
    NAME = 'gdumb'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(GDumb, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def observe(self, inputs, labels, not_aug_inputs):
        # # for dropping test
        inputs, labels, not_aug_inputs, _ = self.drop_unlabeled(inputs, labels, not_aug_inputs)
        if len(inputs) == 0:
            return 0

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)
        return 0


    def end_task(self, dataset):
        # new model
        self.net = dataset.get_backbone().to(self.device)
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        fit_buffer(self, 3000)