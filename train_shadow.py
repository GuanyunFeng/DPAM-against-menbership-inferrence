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
import argparse
import numpy as np

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

matplotlib.use('Agg')

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--dataset', type=str, default="MNIST", metavar='N',
                    help='classification dataset(default: MNIST)')
parser.add_argument('--train_size', type=int, default=2500, metavar='N',
                    help='train data size for target model(default: 2500)')
parser.add_argument('--valid_size', type=int, default=2000, metavar='N',
                    help='valid data size for target model(default: 2500)')
parser.add_argument('--attack_size', type=int, default=2500, metavar='N',
                    help='attack data size for attack model(default: 2500)')
parser.add_argument('--shadow_num', type=int, default=10, metavar='N',
                    help='number of shadow model(default: 25)')
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
parser.add_argument('--result_dir', type=str, default="./result", metavar='N',
                    help='result dir (default: result)')
args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda:0" if args.cuda else "cpu")


def get_labels(Net_G, optimizer_G, dataloader):
    X_train_h = []
    X_train = []
    C_train = []
    X_val_h = []
    X_val = []
    Y_val = []
    C_val = []
    X_attack_h = []
    X_attack = []
    Y_attack = []
    C_attack = []
    for phase in ['train', 'val', 'attack']:
        # Iterate over data.
        for data, target in dataloader[phase]:
            inputs, labels = data.to(device), target.to(device)
            # forward
            outputs_G = Net_G(inputs.float())

            length = outputs_G[0].cpu().detach().numpy().shape[0]
            out = outputs_G[0].cpu().detach().numpy()
            h = outputs_G[1].cpu().detach().numpy()
            for i in range(length):
                if phase == "train":
                    X_train.append(out[i])
                    X_train_h.append(h[i])
                    X_val.append(out[i])
                    X_val_h.append(h[i])
                    X_attack.append(out[i])
                    X_attack_h.append(h[i])
                    Y_val.append(1.)
                    Y_attack.append(1.)
                elif phase == "val":
                    X_val.append(out[i])
                    X_val_h.append(h[i])
                    Y_val.append(0.)
                else:
                    X_attack.append(out[i])
                    X_attack_h.append(h[i])
                    Y_attack.append(0.)
            if phase == "train":
                for cla in labels.cpu().detach().numpy():
                    C_train.append(cla)
                    C_val.append(cla)
                    C_attack.append(cla)
            elif phase == "val":
                for cla in labels.cpu().detach().numpy():
                    C_val.append(cla)
            else:
                for cla in labels.cpu().detach().numpy():
                    C_attack.append(cla)
    X_train,X_train_h, C_train = shuffle(np.array(X_train),np.array(X_train_h), np.array(C_train),  random_state=0)
    X_val, X_val_h, Y_val,C_val = shuffle(np.array(X_val),np.array(X_val_h), np.array(Y_val).reshape((len(Y_val),1)), np.array(C_val),  random_state=0)
    X_attack, X_attack_h, Y_attack, C_attack = shuffle(np.array(X_attack),np.array(X_attack_h), np.array(Y_attack).reshape((len(Y_attack),1)), np.array(C_attack),  random_state=0)
    X_val, Y_val = torch.Tensor(X_val), torch.Tensor(Y_val)
    X_train = torch.Tensor(X_train)

    return X_train,X_train_h, C_train, X_val, X_val_h, Y_val, C_val, X_attack,X_attack_h, Y_attack, C_attack



def train2(Net_G, optimizer_G, dataloader, para=0):

    dd = [[] for i in range(args.classnum)]
    for data, target in dataloader["train"]:
        inputs, labels = data.to(device), target.to(device)
        for i in range(labels.shape[0]):
            dd[labels[i]].append(inputs[i])
    
    X_attacks = []
    X_attacks_h = []
    Y_attacks = []
    C_attacks = []

    for epoch in range(args.epochs):
        # train net G
        Net_G.train()  # Set model to training mode
            # get batch size
        for it in range(int(args.train_size/args.batch_size)):
            half_batch = int(args.batch_size/2)
            seq = np.random.randint(args.classnum , size= half_batch)
            batch_x = [None for i in range(args.batch_size)]
            for i in range(half_batch):
                l = seq[i]
                ids = np.random.randint(len(dd[l]), size= 2)
                batch_x[i] = dd[l][ids[0]]
                batch_x[i + half_batch] = dd[l][ids[1]]

            seq_tensor = torch.from_numpy(seq)
            batch_y = torch.cat((seq_tensor,seq_tensor),0).to(device)
            batch_x = torch.stack(batch_x, 0)
            
            optimizer_G.zero_grad()
            outputs = Net_G(batch_x.float())
            loss = nn.CrossEntropyLoss()(outputs[0],batch_y)
            loss.backward()
            optimizer_G.step()
        
        _, _, _,_,_, _, _, X_attack,X_attack_h, Y_attack, C_attack = get_labels(Net_G, optimizer_G, dataloader)
        X_attacks.append(X_attack)
        X_attacks_h.append(X_attack_h)
        Y_attacks.append(Y_attack)
        C_attacks.append(C_attack)
            
    return X_attacks, X_attacks_h, Y_attacks, C_attacks



if __name__=="__main__":

    #train shadow model
    print("START TRAINING SHADOW MODEL")
    data_train_sets = [[] for i in range(args.epochs)]
    data_h_train_sets = [[] for i in range(args.epochs)]
    label_train_sets = [[] for i in range(args.epochs)]
    for num_model_shadow in range(args.shadow_num):
        if args.dataset == "MNIST":
            dataloader = load_mnist(args.train_size, args.valid_size, args.attack_size, args.batch_size)
            model_shadow = Net_mnist().to(device)
            args.classnum = 10
        elif args.dataset == "CIFAR10":
            dataloader = load_cifar10(args.train_size, args.valid_size, args.attack_size, args.batch_size)
            model_shadow = Net_cifar10().to(device)
            args.classnum = 10
        elif args.dataset == "CIFAR100":
            dataloader = load_cifar100(args.train_size, args.valid_size, args.attack_size, args.batch_size)
            model_shadow = Net_cifar100().to(device)
            args.classnum = 100
        elif args.dataset == "PURCHASE":
            dataloader = load_PURCHASE(args.train_size, args.valid_size, args.attack_size, args.batch_size)
            model_shadow = Net_purchase100().to(device)
            args.classnum = 100
        elif args.dataset == "TEXAS":
            dataloader = load_TEXAS(args.train_size, args.valid_size, args.attack_size, args.batch_size)
            model_shadow = Net_texas100().to(device)
            args.classnum = 100
        elif args.dataset == "LOCATION":
            dataloader = load_LOCATION(args.train_size, args.valid_size, args.attack_size, args.batch_size)
            model_shadow = Net_location30().to(device)
            args.classnum = 30
        
        optimizer = optim.SGD(model_shadow.parameters(), lr=args.lr, momentum=args.momentum)

        X_shadow, X_shadow_h, Y_shadow, C_shadow = train2(model_shadow, optimizer, dataloader)

        for epoch in range(args.epochs):
            data_train_sets[epoch].append(X_shadow[epoch])
            data_h_train_sets[epoch].append(X_shadow_h[epoch])
            label_train_sets[epoch].append(Y_shadow[epoch])
    
    path = "{}/{}_attack".format(args.result_dir, args.dataset)
    if not os.path.exists(path):
        os.mkdir(path)
    np.save("{}/x_attack.npy".format(path), np.array(data_train_sets))
    np.save("{}/x_h_attack.npy".format(path), np.array(data_h_train_sets))
    np.save("{}/y_attack.npy".format(path), np.array(label_train_sets))
