#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='save',
                        help="dic to save results (ending without /)")
    parser.add_argument('--epochs', type=int, default=200,
                        help="rounds of training")
    parser.add_argument('--num_users', type=int,
                        default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help="the fraction of clients: C")
    parser.add_argument('--malicious', type=float, default=0.4, help="proportion of mailicious clients")
    parser.add_argument('--attack', type=str,
                        default='badnet', help='attack method')
    parser.add_argument('--local_ep', type=int, default=2,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,
                        help="local batch size: B")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument('--model', type=str,
                        default='InsPLAD_Resnet18', help='model name')
    parser.add_argument('--dataset', type=str,
                        default='InsPLAD', help="name of dataset")
    parser.add_argument('--defence', type=str,
                        default='avg', help="strategy of defence")
    parser.add_argument('--attack_begin', type=int, default=50,
                        help="the accuracy begin to attack")

    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID, -1 for CPU")

    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD momentum (default: 0.5)")

    parser.add_argument('--asr_source_class', type=int, default=0, help='source class')
    parser.add_argument('--asr_target_class', type=int, default=1, help='target class')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')

    parser.add_argument('--all_clients', action='store_true',
                        help='aggregation over all clients')
    args = parser.parse_args()
    return args
