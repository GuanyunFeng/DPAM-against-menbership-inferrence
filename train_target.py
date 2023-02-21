import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
#from opacus import PrivacyEngine
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

matplotlib.use('Agg')

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
                    help='learning rate (default: 0.001)')
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

def evaluate(Net_G, optimizer_G, dataloader, datasize):
    corrects1 = 0
    corrects3 = 0
    corrects5 = 0
    corrects10 = 0
    Net_G.eval()
    for data, target in dataloader:
        inputs, labels = data.to(device), target.to(device)
        optimizer_G.zero_grad()
        outputs_G = Net_G(inputs.float())
        _, indices = outputs_G[0].topk(10,dim=1, largest=True, sorted=True)
        indices = indices.t()
        ind_t= indices.eq(labels.data.view(1, -1).expand_as(indices))
        corrects1 += ind_t[:1].reshape(-1).float().sum(0)
        corrects3 += ind_t[:3].reshape(-1).float().sum(0)
        corrects5 += ind_t[:5].reshape(-1).float().sum(0)
        corrects10 += ind_t[:10].reshape(-1).float().sum(0)
    return corrects1/datasize,corrects3/datasize,corrects5/datasize,corrects10/datasize,


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


def train1(Net_G, optimizer_G, Net_D, optimizer_D,dataloader,para=2):
    X_attacks_h = []
    X_attacks = []
    Y_attacks = []
    C_attacks = []
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=100, gamma=0.1, last_epoch=-1)
    dd = [[] for i in range(args.classnum)]
    for data, target in dataloader["train"]:
        inputs, labels = data.to(device), target.to(device)
        for i in range(labels.shape[0]):
            dd[labels[i]].append(inputs[i])
    
    acc_train = []
    acc_val = []

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
            
            train_labels = Variable(torch.ones((batch_x.shape[0],1))).to(device)
            optimizer_G.zero_grad()
            # forward
            with torch.set_grad_enabled(True):
                outputs_G = Net_G(batch_x.float())
                loss_G = nn.CrossEntropyLoss()(outputs_G[0], batch_y)
                
                outputs_D = Net_D(outputs_G[0])
                loss_D = nn.BCELoss()(outputs_D, train_labels)
                loss = loss_G - para*loss_D

                loss.backward()
                optimizer_G.step()

        scheduler.step()
        
        ac_val = evaluate(Net_G, optimizer_G, dataloader["val"], args.valid_size)
        acc_val.append(ac_val[0].cpu().numpy())
        print("val accuracy:{}".format(ac_val))
        ac_train = evaluate(Net_G, optimizer_G, dataloader["train"], args.train_size)
        acc_train.append(ac_train[0].cpu().numpy())
        print("train accuracy:{}".format(ac_train))
        X_train,X_train_h, C_train, X_val,X_val_h, Y_val, C_val,  X_attack,X_attack_h, Y_attack, C_attack = get_labels(Net_G, optimizer_G, dataloader)
        X_attacks.append(X_attack)
        X_attacks_h.append(X_attack)
        Y_attacks.append(Y_attack)
        C_attacks.append(C_attack)

        for i in range(int(len(Y_val)/64)):
            input_x = X_val[i*64:(i+1)*64].to(device)
            input_y = Y_val[i*64:(i+1)*64].to(device)
            optimizer_D.zero_grad()
            outputs_D = Net_D(input_x)
            loss_D = nn.BCELoss()(outputs_D, input_y)
            loss_D.backward()
            optimizer_D.step()
        
    return X_attack,X_attacks_h, Y_attack, C_attack, np.array(acc_train),np.array(acc_val)


def train2(Net_G, optimizer_G, dataloader, para=0):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=25, gamma=0.1, last_epoch=-1)
    
    acc_train = []
    acc_val = []


    for epoch in range(args.epochs):
        # train net G
        Net_G.train()  # Set model to training mode
            # get batch size
        print(epoch)
        for data, target in dataloader["train"]:
            batch_x, batch_y = data.to(device), target.to(device)
            optimizer_G.zero_grad()
            outputs = Net_G(batch_x.float())
            #print(outputs[0].shape)
            #print(outputs[1].shape)
            loss_ce = nn.CrossEntropyLoss()(outputs[0],batch_y)
            outputs2 = torch.chunk(outputs[0], 2, dim=0)
            loss_same = torch.norm(outputs2[0]-outputs2[1],p=2)
            loss = loss_ce+int((epoch)/10)*para*loss_same
            loss.backward()
            optimizer_G.step()
        #scheduler.step()
        
        ac_val = evaluate(Net_G, optimizer_G, dataloader["val"], args.valid_size)
        print("val accuracy:{}".format(ac_val))
        acc_val.append(ac_val[0].cpu().numpy())
        ac_train = evaluate(Net_G, optimizer_G, dataloader["train"], args.train_size)
        acc_train.append(ac_train[0].cpu().numpy())
        print("train accuracy:{}".format(ac_train))
        _,_, _, _,_, _, _, X_attack,X_attack_h, Y_attack, C_attack = get_labels(Net_G, optimizer_G, dataloader)
            
    return X_attack,X_attack_h, Y_attack, C_attack, np.array(acc_train),np.array(acc_val)



if __name__=="__main__":
    print(args)
    
    args.result_dir = "/".join([args.result_dir, args.model])
    print(args.result_dir)
    if args.defense == "None":
        path = "{}/{}_{}".format(args.result_dir,args.dataset, args.defense)
    else:
        path = "{}/{}_{}_{}".format(args.result_dir,args.dataset, args.defense,args.parameter)
    if not os.path.exists(path):
        os.mkdir(path)

    print("load data")
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

    print("begen training")
    #Net_G.load_state_dict(torch.load("{}/weights.pth".format(path)))

    if args.defense == "DPSGD":
        privacy_engine = PrivacyEngine(
            Net_G,
            sample_rate=0.001,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=0.2,
            max_grad_norm=1.0,
            secure_rng=False,
        )
        privacy_engine.attach(optimizer_G)
        X_attack, X_attack_h, Y_attack, C_attack,  acc_train, acc_val = train2(Net_G, optimizer_G, dataloader)

    elif args.defense == "GAN":
        if args.classnum == 10:
            Net_D = Net_discriminator10().to(device)
        elif args.classnum == 30:
            Net_D = Net_discriminator30().to(device)
        elif args.classnum == 100:
            Net_D = Net_discriminator100().to(device)
        optimizer_D = optim.SGD(Net_D.parameters(), lr=args.lr, momentum=args.momentum)
        X_attack, X_attack_h, Y_attack, C_attack,  acc_train, acc_val = train1(Net_G, optimizer_G,Net_D, optimizer_D, dataloader,para=args.parameter)
    
    elif args.defense == "OUR":
        X_attack, X_attack_h, Y_attack, C_attack, acc_train, acc_val = train2(Net_G, optimizer_G, dataloader, para=args.parameter)

    elif args.defense == "None":
        X_attack, X_attack_h, Y_attack, C_attack, acc_train, acc_val = train2(Net_G, optimizer_G, dataloader,para=0)

    np.save("{}/x.npy".format(path), np.array(X_attack))
    print(np.array(X_attack_h).shape)
    np.save("{}/x_h.npy".format(path), np.array(X_attack_h))
    np.save("{}/y.npy".format(path), np.array(Y_attack))
    np.save("{}/c.npy".format(path), np.array(C_attack))
    np.save("{}/accuracy_train.npy".format(path), np.array(acc_train))
    np.save("{}/accuracy_val.npy".format(path), np.array(acc_val))
    torch.save(Net_G.state_dict(), "{}/weights.pth".format(path))

