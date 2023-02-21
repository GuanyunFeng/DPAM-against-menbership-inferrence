import os
import argparse
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
from model import *
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def get_topK(x,k):
    index = np.argpartition(x,-k)[-k:]
    minn = x[index[0]]-1e-10
    x[x< minn] = 0
    return x
    
def to_one_hot(tensor, num_clsses):
    assert tensor.dim() <= 1, "[Error] tensor.dim >= 1"
    one_hot = torch.zeros(len(tensor), num_clsses)
    idx = range(len(tensor))
    one_hot[idx, tensor.reshape(-1, )] = 1
    return one_hot

def attack(x_train,x_train_h, y_train, x_attack,x_h_attack, y_attack, model_name="knn"):
    if model_name == "lgb":
        model = lgb.LGBMClassifier(objective='binary', reg_lambda=10, n_estimators=1000)
    elif model_name == "lr":
        model = LogisticRegression(max_iter=5000)
    elif model_name == "dtc":
        model = tree.DecisionTreeClassifier()
    elif model_name == "rfc":
        model = RandomForestClassifier()
    elif model_name == "knn":
        model = KNeighborsClassifier()
    elif model_name == "mlp":
        model = MLPClassifier()

    accuracy = []
    precise = []
    recall = []

    #for epoch in range(49,len(x_attack),10):
    data_train_set = np.concatenate(x_attack[-1])
    data_h_train_set = np.concatenate(x_h_attack[-1])
    label_train_set = np.concatenate(y_attack[-1])

    index = np.where(label_train_set == 0)[0]
    #num = 5000
    data_nomem_set = data_train_set[index]
    data_h_nomem_set = data_h_train_set[index]
    label_nomem_set = label_train_set[index]

    num = label_nomem_set.shape[0]
    index = np.where(label_train_set == 1)[0]
    data_mem_set = data_train_set[index][:num]
    data_h_mem_set = data_h_train_set[index][:num]
    label_mem_set = label_train_set[index][:num]
    
    data_train_set = np.concatenate([data_mem_set,data_nomem_set])
    data_h_train_set = np.concatenate([data_h_mem_set,data_h_nomem_set])
    label_train_set = np.concatenate([label_mem_set,label_nomem_set])

    data_train_set, label_train_set = shuffle(data_train_set, label_train_set,  random_state=0)

    label_train_set = label_train_set.astype(np.int16)
    data_train_set = data_train_set
    label_train_set = label_train_set

    if model_name != "wb":
        model.fit(data_train_set, label_train_set)
        y_pred = model.predict(x_train)
        y_real = y_train.astype(np.int16)
    else:
        model = Attack_Mlp().to("cuda")
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        for i in range(10):
            for j in range(data_train_set.shape[0]//64):
                optimizer.zero_grad()
                x = torch.from_numpy(data_train_set[j*64:j*64+64]).to("cuda")
                h = torch.from_numpy(data_h_train_set[j*64:j*64+64]).to("cuda")
                y = torch.from_numpy(label_train_set[j*64:j*64+64].astype("int64")).to("cuda")
                outputs = model(x,h)
                one_hot_y = torch.nn.functional.one_hot(y.reshape(-1),2).float()
                loss = nn.CrossEntropyLoss()(outputs,one_hot_y)
                loss.backward()
                optimizer.step()

        #model.fit(data_train_set, label_train_set)
        print("finish_training")
        corrects = 0
        model.eval()
        y_pred = []
        y_real = []
        for j in range(data_train_set.shape[0]//64):
            optimizer.zero_grad()
            x = torch.from_numpy(data_train_set[j*64:j*64+64]).to("cuda")
            h = torch.from_numpy(data_h_train_set[j*64:j*64+64]).to("cuda")
            y = torch.from_numpy(label_train_set[j*64:j*64+64].astype("int64")).to("cuda")
            _, outputs = torch.max(model(x,h),1)
            y = y.reshape(-1)
            y_pred.append(outputs.detach().cpu().numpy())
            y_real.append(y.detach().cpu().numpy())
            corrects += torch.sum((outputs == y))
        print(corrects/(data_train_set.shape[0]//64)/64)
        #x_train = x_train[epoch]
        #for i in range(x_train.shape[0]):
        #    x_train[i] = get_topK(x_train[i],3)
        y_pred = np.concatenate(np.array(y_pred))
        y_real = np.concatenate(np.array(y_real))
    #y_pred = model.predict(x_train)
    #y_real = y_train.astype(np.int16)
    precision_general, recall_general, _, _ = precision_recall_fscore_support(y_pred=y_pred, y_true=y_real, average = "macro")
    accuracy_general = accuracy_score(y_true=y_real, y_pred=y_pred)
    print("{} precision:{},recall:{},f1-score:{},accuracy:{}".format(model_name, precision_general,recall_general,2*precision_general*recall_general/(precision_general+recall_general),accuracy_general))
    accuracy.append(accuracy_general)
    precise.append(precision_general)
    recall.append(precision_general)

    return accuracy, precise, recall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ATTACK MNIST')
    parser.add_argument('--dataset', type=str, default="MNIST", metavar='N',
                        help='classification dataset(default: MNIST)')
    parser.add_argument('--defense', type=str, default="None", metavar='N',
                        help='defense(None, DPSGD, GAN, OUR)')
    parser.add_argument('--model', type=str, default="CNN", metavar='N',
                    help='model(CNN, VGG, DenseNet)')
    parser.add_argument('--attack', type=str, default="knn", metavar='N',
                    help='model(CNN, VGG, DenseNet)')
    parser.add_argument('--result_dir', type=str, default="result", metavar='S',
                    help='result dir (default: result)')
    parser.add_argument('--parameter', type=float, default=2.0, metavar='N',
                    help='defense parameter(DPSGD 0.1-1, GAN 1-10, OUR 0.001-0.015)')
    args = parser.parse_args()

    #args.result_dir = "/".join([args.result_dir, args.model])

    if args.defense == "None":
        target_path = "{}/{}/{}_{}/".format(args.result_dir,args.model, args.dataset, args.defense)
    else:
        target_path = "{}/{}/{}_{}_{}/".format(args.result_dir,args.model, args.dataset, args.defense, args.parameter)
    attack_path = "{}/{}_attack/".format(args.result_dir, args.dataset)
    x_train,x_train_h, y_train, c_train = np.load(target_path+"x.npy"),np.load(target_path+"x_h.npy"),np.load(target_path+"y.npy"),np.load(target_path+"c.npy")
    x_attack,x_attack_h, y_attack = np.load(attack_path+"x_attack.npy"),np.load(attack_path+"x_h_attack.npy"),np.load(attack_path+"y_attack.npy")
    for at in ["knn","dtc","rfc","lgb","lr","mlp"]:
        accuracy, precise, recall=attack(x_train,x_train_h,y_train, x_attack,x_attack_h, y_attack, model_name=args.attack)
        #accuracy, precise, recall, accuracy_class, precise_class, recall_class = attack(x_train, y_train, c_train, x_attack, y_attack, model_name=at, num_class=30)
        if not os.path.exists(target_path+args.attack):
            os.mkdir(target_path+args.attack)
        np.save("{}{}/accuracy.npy".format(target_path, args.attack), accuracy)
        np.save("{}{}/precise.npy".format(target_path, args.attack), precise)
        np.save("{}{}/recall.npy".format(target_path, args.attack), recall)

