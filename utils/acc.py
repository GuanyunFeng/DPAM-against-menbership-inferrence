import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
matplotlib.use('agg')

p = [0.145275, 0.2049, 0.25025, 0.28775, 0.319825, 0.348275, 0.3738, 0.39605, 0.418625, 0.437725, 0.45615, 0.473525, 0.48955, 0.505275, 0.52025, 0.53345, 0.54665, 0.5593, 0.57175, 0.583275, 0.594325, 0.6054, 0.615975, 0.62585, 0.63575, 0.64575, 0.655325, 0.664275, 0.67305, 0.68275, 0.691475, 0.69935, 0.70745, 0.71595, 0.723625, 0.7313, 0.73885, 0.74665, 0.754175, 0.76135, 0.768775, 0.775475, 0.78145, 0.7881, 0.7947, 0.8008, 0.80715, 0.81335, 0.818675, 0.824525, 0.830625, 0.835925, 0.841275, 0.8469, 0.8517, 0.857075, 0.86215, 0.86705, 0.872275, 0.87705, 0.8807, 0.8861, 0.889875, 0.894275, 0.898175, 0.90235, 0.9063, 0.909675, 0.91315, 0.916875, 0.91965, 0.923125, 0.926225, 0.929475, 0.932675, 0.9357, 0.93925, 0.94215, 0.9452, 0.9478, 0.950725, 0.95315, 0.955375, 0.957925, 0.960275, 0.962525, 0.964725, 0.966575, 0.9686, 0.970375, 0.972125, 0.9739, 0.975525, 0.977275, 0.9787, 0.979925, 0.98135, 0.98255, 0.9837, 0.984875, 0.985925, 0.986975, 0.987875, 0.988575, 0.9897, 0.990625, 0.991525, 0.9923, 0.992875, 0.993425, 0.993975, 0.994475, 0.99515, 0.9957, 0.996075, 0.996625, 0.99705, 0.997375, 0.9976, 0.997925, 0.998225, 0.998525, 0.998675, 0.99895, 0.99905, 0.999175, 0.999275, 0.999425, 0.999575, 0.9996, 0.999675, 0.9997, 0.9998, 0.99985, 0.999875, 0.999925, 0.999975, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def get_topK(x,k):
    #index = np.argpartition(x,-k)[-k:]
    #minn = x[index[0]]-1e-10
    #x[x< minn] = 0
    return x

def PPP(x, y, c):
    x_last = x[:,:]
    y_last = y[:,:]
    c_last = c[:]
    x_last = nn.Softmax(dim=1)(torch.from_numpy(x_last)).numpy()

    index = np.where(y_last == 0)[0]
    x_last_valid = x_last[index]
    c_last_valid = c_last[index]
    index = np.where(y_last == 1)[0]
    x_last_train = x_last[index]
    c_last_train = c_last[index]

    valid_size = 1000
    x_last_train = x_last_train[:valid_size]
    c_last_train = c_last_train[:valid_size]
    x_last_valid = x_last_valid[:valid_size]
    c_last_valid = c_last_valid[:valid_size]

    #x_last_train = np.round(x_last_train,decimals=1)
    #x_last_valid = np.round(x_last_valid,decimals=1)


    th_acc = 0
    #遍历训练集
    for i in range(x_last_train.shape[0]):
        print("train set:{}".format(i))
        mem_num = 0
        nomem_num = 0
        for j in range(valid_size):
            x1 = get_topK(x_last_train[i],2)
            a = int(np.linalg.norm(x1 - get_topK(x_last_train[j],2))*100)
            mem_num += 1-p[a]
            a = int(np.linalg.norm(x1 - get_topK(x_last_valid[j],2))*100)
            nomem_num += 1-p[a]
        th_acc += mem_num/(mem_num+nomem_num)
        print("mem_num:{} nomem_num:{}".format(mem_num,nomem_num))
    for i in range(x_last_valid.shape[0]):
        print("valid set:{}".format(i))
        mem_num = 0
        nomem_num = 0
        for j in range(valid_size):
            x1 = get_topK(x_last_valid[i],2)
            a = int(np.linalg.norm(x1 - get_topK(x_last_train[j],2))*100)
            mem_num += 1-p[a]
            a = int(np.linalg.norm(x1 - get_topK(x_last_valid[j],2))*100)
            nomem_num += 1-p[a]
        th_acc += nomem_num/(mem_num+nomem_num)
        print("mem_num:{} nomem_num:{}".format(mem_num,nomem_num))
    
    th_acc /= 2*valid_size
    print(th_acc)
    return th_acc

x = np.load("./result/CNN/TEXAS_OUR_0.01/x.npy")
y = np.load("./result/CNN/TEXAS_OUR_0.01/y.npy")
c = np.load("./result/CNN/TEXAS_OUR_0.01/c.npy")
PPP(x,y,c)

'''
x = np.load("./result/CNN/PURCHASE_GAN_2.0/x.npy")
y = np.load("./result/CNN/PURCHASE_GAN_2.0/y.npy")
c = np.load("./result/CNN/PURCHASE_GAN_2.0/c.npy")
PPP(x,y,c,0.2)
'''
'''
x = np.load("../result/CNN/PURCHASE_None/x.npy")
y = np.load("../result/CNN/PURCHASE_None/y.npy")
c = np.load("../result/CNN/PURCHASE_None/c.npy")
PPP(x,y,c,0.5)
'''
