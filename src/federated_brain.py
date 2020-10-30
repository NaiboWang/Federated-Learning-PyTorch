#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, modelC, CNNCifaro, ModerateCNN
from utils import get_dataset, average_weights, exp_details, additive_secret_sharing

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        print("args.gpu", args.gpu)
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)
    # user_groups: dict, 100个user，key是0-100，value为一个数组，组内有600个索引值（对于mnist来说），索引值对应mnist数组中的数据，根据non-iid或iid的不同来得到不同的索引
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = ModerateCNN(args=args)
            # global_model = CNNCifaro(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if args.parallel:
        global_model = torch.nn.DataParallel(global_model)
    global_model.to(device)
    # Set model to training mode
    global_model.train()
    print(global_model)

    # copy weights
    # state_dict() returns a dictionary containing a whole state of the module
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    val_loss_pre, counter = 0, 0
    version_matrix = np.zeros((args.num_users, args.num_users))  # 模型版本矩阵
    models = {}  # 用于存放各个client的模型
    for i in range(args.num_users):
        models[i] = copy.deepcopy(global_model)
        models[i].train()
    # print(models)
    # tqdm进度条功能 progress bar
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch + 1} |\n')

        global_model.train()

        for r in range(args.num_users):
            m = max(int(args.frac * args.num_users), 1)  # 从num_users个user中随机选取frac部分的用户用于训练
            idxs_users = np.random.choice(range(args.num_users), m, replace=False)
            print("Users selected:", idxs_users)
            for idx in idxs_users:
                local_model = LocalUpdate(args=args, dataset=train_dataset,
                                          idxs=user_groups[idx], logger=logger)
                w, loss, t_model = local_model.update_weights(
                    model=copy.deepcopy(models[idx]), global_round=epoch)
                # print("local losses:",loss)
                models[idx].load_state_dict(w)
                version_matrix[idx, idx] = version_matrix[idx, idx] + 1

            idx_user = np.random.choice(range(args.num_users), 1, replace=False)[0]
            v_old = np.reshape(version_matrix[idx_user, :], -1)
            v_new = np.zeros(args.num_users)
            for i in range(args.num_users):
                v_new[i] = version_matrix[i, i]
            # 模型聚合
            w_avg = copy.deepcopy(models[idx_user].state_dict())
            n_participants = 1  # 记录参与的模型总数
            for i in range(args.num_users):
                if v_new[i] > v_old[i]:
                    # print("Averaging Model:",i)
                    version_matrix[idx_user, i] = v_new[i]
                    n_participants = n_participants + 1  # 更新长度
                    w_model_to_merge = copy.deepcopy(models[i].state_dict())
                    for key in w_avg.keys():
                        w_avg[key] = additive_secret_sharing(w_avg[key], w_model_to_merge[key],args)
            for key in w_avg.keys():
                w_avg[key] = torch.true_divide(w_avg[key], n_participants)
            print("Select user:", idx_user, ", total number of participants:", n_participants, ", process:", r + 1, "/",
                  args.num_users)
            global_model.load_state_dict(w_avg)
            # # 训练模型
            # w, loss, t_model = local_model.update_weights(model=copy.deepcopy(global_model), global_round=epoch)
            # print("loss:",loss)
            # 更新版本
            version_matrix[idx_user, idx_user] = version_matrix[idx_user, idx_user] + 1
            models[idx_user].load_state_dict(w_avg)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx_user], logger=logger)
            w, loss, t_model = local_model.update_weights(
                model=copy.deepcopy(models[idx_user]), global_round=epoch)
            print("losses:", loss)

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            local_weights = []
            for i in range(args.num_users):
                v_new[i] = version_matrix[i, i]
            print("versions:", v_new)
            for i in range(args.num_users):
                local_weights.append(models[i].state_dict())
            # update global weights，这里的global_model只是取被选择的local_model的平均值
            global_weights = average_weights(local_weights, args)
            # update global weights
            global_model.load_state_dict(global_weights)
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            test_acc, test_loss = test_inference(args, global_model, train_dataset)
            print("test accuracy for training set: {} after {} epochs\n".format(test_acc, epoch + 1))
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print("test accuracy for test set: {} after {} epochs\n".format(test_acc, epoch + 1))

    for i in range(args.num_users):
        local_weights.append(models[i].state_dict())
    # update global weights，这里的global_model只是取被选择的local_model的平均值
    global_weights = average_weights(local_weights, args)
    # update global weights
    global_model.load_state_dict(global_weights)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    print("file_name:", file_name)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

    # # PLOTTING (optional)
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('Agg')
    # print("Start Plot")
    # # Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    # plt.show()
    # # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    # plt.show()
