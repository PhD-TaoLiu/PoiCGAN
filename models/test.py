#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(iter, net_g, datatest, dataset_source_test, args):
    net_g.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    data_source_loader = DataLoader(dataset_source_test, batch_size=args.bs)
    correct_source = 0
    asr_class_total = len(data_source_loader.dataset)
    for idx, (data, target) in enumerate(data_source_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct_source += (y_pred == args.asr_target_class).sum().item()

    asr_class_target_class = correct_source
    class_asr = 100.00 * correct_source / len(data_source_loader.dataset)
    if args.dataset == 'InsPLAD':
        num_classes = 4

    confusion_matrix = torch.zeros(num_classes, num_classes)
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
        non_mask = (target != args.asr_source_class)
        if non_mask.sum().item() == 0:
            continue
        data = data[non_mask]
        target = target[non_mask]
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=False)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        for t, p in zip(target.view(-1), y_pred.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1
            total_per_class[t.long()] += 1
            if t == p:
                correct_per_class[t.long()] += 1
    len_data_loader = len(data_loader.dataset)
    if args.malicious != 0:
        len_data_loader -= 1
    test_loss /= len_data_loader
    accuracy = 100.00 * correct / len_data_loader
    return accuracy, test_loss, class_asr
