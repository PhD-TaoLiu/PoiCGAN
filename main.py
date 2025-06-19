#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18

from models.Fed import FedAvg
from models.Update import LocalUpdate
from models.test import test_img
from utils.info import print_exp_details, write_info_to_accfile
from utils.options import args_parser

matplotlib.use('Agg')


def write_file(filename, accu_list, args, asr_list, analyse=False):
    write_info_to_accfile(filename, args)
    f = open(filename, "a")
    f.write("main_task_accuracy=")
    f.write(str(accu_list))
    f.write('\n')
    f.write("asr=")
    f.write(str(asr_list))
    f.write('\n')
    f.close()


def write_file_test(filename, acc_train, acc_test):
    f = open(filename, "a")
    f.write("Training accuracy=")
    f.write(str(acc_train))
    f.write('\n')
    f.write("Testing accuracy=")
    f.write(str(acc_test))
    f.write('\n')
    f.close()


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(
        args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    test_mkdir('./' + args.save)
    print_exp_details(args)

    if args.dataset == 'InsPLAD':
        trans_insplad = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        data_dir = "datasets/InsPLAD"
        dataset_train = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=trans_insplad)
        dataset_test = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=trans_insplad)
        dataset_source_test = datasets.ImageFolder(root=os.path.join(data_dir, "bird-nest"), transform=trans_insplad)
        dict_users = np.load('datasets/iid_InsPLAD.npy', allow_pickle=True).item()

        data_dir_gan = "gan/InsPLAD_gan_images"
        dataset_train_gan = datasets.ImageFolder(root=os.path.join(data_dir_gan, "train"), transform=trans_insplad)
        dict_dir_gan = "gan/gan_InsPLAD.npy"
        dict_users_gan = np.load(dict_dir_gan, allow_pickle=True).item()
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'InsPLAD_Resnet18':
        net_glob = resnet18(num_classes=4)
        num_ftrs = net_glob.fc.in_features
        net_glob.fc = torch.nn.Linear(num_ftrs, 4)
        net_glob = net_glob.to(args.device)
    else:
        exit('Error: unrecognized model')

    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    base_info = '{}_{}'.format(args.dataset, int(time.time()))
    filename = './' + args.save + '/accuracy_file_{}.txt'.format(base_info)

    val_acc_list = [0]
    asr_list = [0]

    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]

    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)

        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        idxs_users_mal = []
        if args.malicious * m % 1 == 0:
            idxs_users_mal = np.random.choice(range(int(args.malicious * args.num_users)), int(args.malicious * m),
                                              replace=False)
        else:
            if iter % 2 == 0:
                idxs_users_mal = np.random.choice(range(int(args.malicious * args.num_users)), int(args.malicious * m),
                                                  replace=False)
            else:
                idxs_users_mal = np.random.choice(range(int(args.malicious * args.num_users)),
                                                  int(args.malicious * m) + 1, replace=False)
        mal_idx = 0
        if iter >= args.attack_begin:
            attack_number = int(args.malicious * m)
            if args.malicious * m % 1 != 0 and iter % 2 != 0:
                attack_number += 1
        else:
            attack_number = 0
        for num_turn, idx in enumerate(idxs_users):
            if attack_number > 0:
                attack = True
            else:
                attack = False
            if attack == True:
                if args.attack == "badnet":
                    local = LocalUpdate(
                        args=args, dataset=dataset_train, idxs=dict_users[idx], malicious=True,
                        dataset_mal=dataset_train_gan, idxs_mal=dict_users_gan[idxs_users_mal[mal_idx]])
                    w, loss = local.train(
                        net=copy.deepcopy(net_glob).to(args.device))
                attack_number -= 1
                mal_idx += 1

            else:
                local = LocalUpdate(
                    args=args, dataset=dataset_train, idxs=dict_users[idx], malicious=False)
                w, loss = local.train(
                    net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        if args.defence == 'avg':
            w_glob = FedAvg(w_locals)
        else:
            print("Wrong Defense Method")
            os._exit(0)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

        if iter % 1 == 0:
            acc_test, _, class_asr = test_img(iter, net_glob, dataset_test, dataset_source_test, args)
            print("Main accuracy: {:.2f}".format(acc_test))
            print("ASR of class {} - {} : {:.2f}".format(args.asr_source_class, args.asr_target_class, class_asr))
            val_acc_list.append(acc_test.item())
            asr_list.append(class_asr)
            write_file(filename, val_acc_list, args, asr_list)

    # plot loss curve
    plt.figure()
    plt.xlabel('communication')
    plt.ylabel('rate')
    plt.plot(val_acc_list, label='acc')
    plt.plot(asr_list, label='asr')
    plt.legend()
    title = base_info
    plt.title(title)
    plt.savefig('./' + args.save + '/' + title + '.pdf', format='pdf', bbox_inches='tight')
