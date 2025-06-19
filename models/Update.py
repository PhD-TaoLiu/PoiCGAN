#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, malicious=False, dataset_mal=None, idxs_mal=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if malicious:
            if args.attack == "badnet":
                self.ldr_train = DataLoader(DatasetSplit_mal(
                    dataset_mal, idxs_mal, args), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(
                dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        self.model = args.model

    def train(self, net):
        net.train()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class DatasetSplit_mal(Dataset):
    def __init__(self, dataset, idxs, args):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.asr_target_class = args.asr_target_class

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, self.asr_target_class