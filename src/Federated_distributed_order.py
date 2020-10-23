#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

"""
有序的从1到num_users训练model，第i个model初始化为第i-1个model训练后的模型

与所有用户把数据交给center分批训练的唯一区别就在于互相看不到对方的数据

"""

import os
import copy
import time
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, modelC, CNNCifaro, ModerateCNN
from utils import get_dataset, average_weights, exp_details

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        print("args.gpu",args.gpu)
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

    global_model = nn.DataParallel(global_model)
    global_model = global_model.to(device)
    # Set the model to train and send it to device.
    # global_model.to(device)
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
    print_every = 5
    val_loss_pre, counter = 0, 0
    # tqdm进度条功能 progress bar
    for epoch in tqdm(range(args.epochs)):
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train() # 设置成训练模式
        idxs_users = range(args.num_users)

        for idx in idxs_users:
            print("Training at user %d/%d." % (idx+1,args.num_users))
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, global_model = local_model.update_weights(
                model=global_model, global_round=epoch)

            # update global weights将下个模型要用的模型改成上一个模型的初始值
            # global_model.load_state_dict(w)

        # loss_avg = sum(local_losses) / len(local_losses)
        # train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[idx], logger=logger) # 只是返回了local_model的类
        #     acc, loss = local_model.inference(model=global_model) # 这一步只是用了local_model的数据集，即用global_model在training dataset上做测试
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            # print(f'Training Loss : {np.mean(np.array(train_loss))}')
            # print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            test_acc, test_loss = test_inference(args, global_model, train_dataset)
            print("test accuracy for training set: {} after {} epochs\n".format(test_acc, epoch + 1))
            test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print("test accuracy for test set: {} after {} epochs\n".format(test_acc, epoch + 1))

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)
    print("file_name:",file_name)
    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

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
