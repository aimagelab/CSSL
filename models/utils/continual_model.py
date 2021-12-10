# Copyright 2021-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim import SGD
import torch
import torchvision
from argparse import Namespace
from utils.conf import get_device


def extract_features(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)

    return x


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME = None
    COMPATIBILITY = []

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: torchvision.transforms) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        self.device = get_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        pass

    @staticmethod
    def discard_unsupervised_labels(inputs, labels, not_aug_inputs):
        mask = labels < 1000

        return inputs[mask], labels[mask], not_aug_inputs[mask]

    @staticmethod
    def discard_supervised_labels(inputs, labels, not_aug_inputs):
        mask = labels >= 1000

        return inputs[mask], labels[mask], not_aug_inputs[mask]

    def guess_notaug_weighted(self, inputs, labels, not_aug_inputs):
        mask = labels > 999
        if not self.buffer.is_empty():
            if mask.sum():
                with torch.no_grad():
                    cur_feats = extract_features(self.feature_extractor,
                                                 torch.stack([self.norm_transform(ee) for ee in not_aug_inputs]))
                _, buf_labels, buf_feats = self.buffer.get_data(100)
                buf_feats = buf_feats.unsqueeze(1)
                dists = - (buf_feats - cur_feats).pow(2).sum(2)
                soft_dists = torch.softmax((dists - dists.mean(0)) / dists.std(0), dim=0)
                lab = self.eye[buf_labels].unsqueeze(1) * soft_dists.unsqueeze(2)
                labels[mask] = lab.mean(0).max(dim=1)[1][mask]
                assert (labels < 999).all()
        else:
            not_aug_inputs = not_aug_inputs[labels < 999]
            inputs = inputs[labels < 999]
            labels = labels[labels < 999]
            if inputs.shape[0]:
                with torch.no_grad():
                    cur_feats = extract_features(self.feature_extractor,
                                                 torch.stack([self.norm_transform(ee) for ee in not_aug_inputs]))
            else:
                cur_feats = inputs
        return inputs, labels, not_aug_inputs, cur_feats

    def pseudo_label(self, inputs, labels, not_aug_inputs, conf=5.5):
        self.net.eval()
        with torch.no_grad():
            psi_outputs = self.net(inputs)
            
            confs = psi_outputs[:, self.cpt * self.task: self.cpt * (self.task+1)].topk(2, axis=1)[0]
            confs = confs[:, 0] - confs[:, 1]
            conf_thresh = conf
            confidence_mask = confs > conf_thresh #torch.zeros_like(labels).bool()
            _, psi_labels = torch.max(psi_outputs.data[:, self.cpt * self.task: self.cpt * (self.task+1)], 1)
            psi_labels += self.cpt * self.task
            
            out_labels = labels.clone()
            if confidence_mask.sum():
                out_labels[(labels > 999) & confidence_mask] = psi_labels[(labels > 999) & confidence_mask]
        self.net.train()
        return self.drop_unlabeled(inputs, out_labels, not_aug_inputs)[:-1]

    def guess_notaug(self, labels, not_aug_inputs):
        if (labels > 999).sum():
            extract_features(self.feature_extractor, not_aug_inputs)
            feats = self.buffer.logits.unsqueeze(1)

        labels = self.eye[self.buffer.labels[(self.class_means - feats).pow(2).sum(2).topk(
            self.args.k, largest=False)[1]].mode()[0]]
        return labels

    def drop_unlabeled(self, inputs, labels, not_aug_inputs):
        not_aug_inputs = not_aug_inputs[labels < 1000]
        inputs = inputs[labels < 1000]
        labels = labels[labels < 1000]
        if inputs.shape[0] and hasattr(self, 'feature_extractor'):
            with torch.no_grad():
                cur_feats = extract_features(self.feature_extractor,
                                             torch.stack([self.norm_transform(ee) for ee in not_aug_inputs]))
        else:
            cur_feats = inputs
        return inputs, labels, not_aug_inputs, cur_feats
