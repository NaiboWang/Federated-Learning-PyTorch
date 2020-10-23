#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from utils import get_dataset
from options import args_parser
from update import test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, modelC, CNNCifaro

if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        print("args.gpu", args.gpu)
        torch.cuda.set_device(int(args.gpu))
        print("using GPU")
    device = 'cuda' if args.gpu else 'cpu'

    # load datasets
    train_dataset, test_dataset, _ = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
            # global_model = modelC(args=args)
    elif args.model == 'mlp':
        # Multi-layer perceptron
        img_size = train_dataset[0][0].shape # [1,28,28] if mnist
        len_in = 1
        for x in img_size: # x will be 1,28,28 if mnist
            len_in *= x
        global_model = MLP(dim_in=len_in, dim_hidden=64,
                           dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model = torch.nn.DataParallel(global_model)
    global_model.to(device)
    # Set model to training mode
    global_model.train()
    print(global_model)

    # Training
    # Set optimizer and criterion
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(global_model.parameters(), lr=args.lr,
                                     weight_decay=1e-4)
        print("Using Adam")

    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss)/len(batch_loss)
        print('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
        # Train Accuracy
        train_acc, train_loss = test_inference(args, global_model, train_dataset)
        print('Training on', len(train_dataset), 'samples')
        print("Training Accuracy: {:.2f}% for epoch {}".format(100 * train_acc,epoch))
        # testing
        test_acc, test_loss = test_inference(args, global_model, test_dataset)
        print('Test on', len(test_dataset), 'samples')
        print("Test Accuracy: {:.2f}% for epoch {}".format(100 * test_acc,epoch))

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.show()
    plt.savefig('../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                 args.epochs))


