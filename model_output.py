import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
from opacus import PrivacyEngine
from dataloaders import *
from model import *
from densenet import DenseNet
import argparse
import numpy as np
import random

from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib
import time
import copy
import math

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', type=str, default="MNIST", metavar='N',
                    help='classification dataset(default: MNIST)')
parser.add_argument('--defense', type=str, default="None", metavar='N',
                    help='defense(None, DPSGD, GAN, OUR)')
parser.add_argument('--parameter', type=float, default=2.0, metavar='N',
                    help='defense parameter(DPSGD 0.1-1, GAN 1-10, OUR 0.001-0.015)')
parser.add_argument('--model', type=str, default="CNN", metavar='N',
                    help='model(CNN, VGG, DenseNet)')                  
parser.add_argument('--train_size', type=int, default=2500, metavar='N',
                    help='train data size for target model(default: 2500)')
parser.add_argument('--valid_size', type=int, default=2000, metavar='N',
                    help='valid data size for target model(default: 2500)')
parser.add_argument('--attack_size', type=int, default=2500, metavar='N',
                    help='attack data size for attack model(default: 2500)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 25)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enable CUDA training')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--result_dir', type=str, default="./result", metavar='S',
                    help='result dir (default: result)')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda:1" if args.cuda else "cpu")

def get_labels(Net_G, dataloader):
    Net_G.eval()
    X_attack = []
    Y_attack = []
    C_attack = []
    for phase in ['train', 'val', 'attack']:
        # Iterate over data.
        for data, target in dataloader[phase]:
            inputs, labels = data.to(device), target.to(device)
            # forward
            outputs_G = Net_G(inputs.float())

            for out in outputs_G.cpu().detach().numpy():
                if phase == "train":
                    X_attack.append(out)
                    Y_attack.append(1.)
                elif phase == "attack":
                    X_attack.append(out)
                    Y_attack.append(0.)
            if phase == "train" or phase == "attack":
                for cla in labels.cpu().detach().numpy():
                    C_attack.append(cla)
    X_attack, Y_attack, C_attack = np.array(X_attack), np.array(Y_attack).reshape((len(Y_attack),1)), np.array(C_attack)
    return X_attack, Y_attack, C_attack

if __name__=="__main__":
    args.result_dir = "/".join([args.result_dir, args.model])
    print(args.result_dir)
    path = "{}/{}_{}_{}".format(args.result_dir,args.dataset, args.defense,args.parameter)

    if args.dataset == "MNIST":
        dataloader = load_mnist(args.train_size, args.valid_size, args.attack_size, args.batch_size)
        Net_G = Net_mnist().to(device)
        optimizer_G = optim.SGD(Net_G.parameters(), lr=args.lr, momentum=args.momentum)
        args.classnum = 10
    
    elif args.dataset == "CIFAR10":
        dataloader = load_cifar10(args.train_size, args.valid_size, args.attack_size, args.batch_size)
        if args.model == "DenseNet":
            Net_G = DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10).to(device)
        else:
            Net_G = Net_cifar10().to(device)
        optimizer_G = optim.SGD(Net_G.parameters(), lr=args.lr, momentum=args.momentum)
        args.classnum = 10
    
    elif args.dataset == "CIFAR100":
        dataloader = load_cifar100(args.train_size, args.valid_size, args.attack_size, args.batch_size)
        if args.model == "DenseNet":
            Net_G = DenseNet(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=100).to(device)
        else:
            Net_G = Net_cifar100().to(device)
        optimizer_G = optim.SGD(Net_G.parameters(), lr=args.lr, momentum=args.momentum)
        args.classnum = 100

    elif args.dataset == "PURCHASE":
        dataloader = load_PURCHASE(args.train_size, args.valid_size, args.attack_size, args.batch_size)
        Net_G = Net_purchase100().to(device)
        optimizer_G = optim.SGD(Net_G.parameters(), lr=args.lr, momentum=args.momentum)
        args.classnum = 100
    
    elif args.dataset == "TEXAS":
        dataloader = load_TEXAS(args.train_size, args.valid_size, args.attack_size, args.batch_size)
        Net_G = Net_texas100().to(device)
        optimizer_G = optim.SGD(Net_G.parameters(), lr=args.lr, momentum=args.momentum)
        args.classnum = 100

    elif args.dataset == "LOCATION":
        dataloader = load_LOCATION(args.train_size, args.valid_size, args.attack_size, args.batch_size)
        Net_G = Net_location30().to(device)
        optimizer_G = optim.SGD(Net_G.parameters(), lr=args.lr, momentum=args.momentum)
        args.classnum = 30

    Net_G.load_state_dict(torch.load("{}/weights.pth".format(path)))
    x,y,c = get_labels(Net_G, dataloader)
    np.save("{}/x.npy".format(path), np.array(x))
    np.save("{}/y.npy".format(path), np.array(y))
    np.save("{}/c.npy".format(path), np.array(c))
