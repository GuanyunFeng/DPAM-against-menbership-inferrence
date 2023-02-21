import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import math
matplotlib.use('agg')

def PP(x, y, c, name):
    x_last = x[-1,:,:]
    y_last = y[-1,:,:]
    c_last = c[-1,:]
    x_last = nn.Softmax(dim=1)(torch.from_numpy(x_last)).numpy()

    index = np.where(y_last == 0)[0]
    x_last_valid = x_last[index]
    c_last_valid = c_last[index]
    index = np.where(y_last == 1)[0]
    x_last_train = x_last[index]
    c_last_train = c_last[index]

    gen_error = [0 for i in range(100)]

    for class_index in range(100):
        index = np.where(c_last_train== class_index)[0]
        x_train = x_last_train[index]
        train_res = np.argmax(x_train,axis = 1)
        train_acc = train_res[train_res==class_index].shape[0]/train_res.shape[0]

        index = np.where(c_last_valid== class_index)[0]
        x_valid = x_last_valid[index]
        valid_res = np.argmax(x_valid,axis = 1)
        valid_acc = valid_res[valid_res==class_index].shape[0]/valid_res.shape[0]
        
        gen_error[class_index] = abs(train_acc - valid_acc)

    gen_error.sort()
    print(gen_error)
    y = [(i+1)/len(gen_error) for i in range(len(gen_error))]
    y.append(1)
    gen_error.append(0.85)
    plt.plot(gen_error,y,label=name)

def PPP(x, y, c, name):
    x_last = x[-1,:,:]
    y_last = y[-1,:,:]
    c_last = c[-1,:]
    x_last = nn.Softmax(dim=1)(torch.from_numpy(x_last)).numpy()

    index = np.where(y_last == 0)[0]
    x_last_valid = x_last[index]
    c_last_valid = c_last[index]
    index = np.where(y_last == 1)[0]
    x_last_train = x_last[index]
    c_last_train = c_last[index]

    entropy_trains = [0 for i in range(10)]
    entropy_valid = [0 for i in range(10)]


    x_train = x_last_train
    x_valid = x_last_valid

    for i in range(x_train.shape[0]):
        entropy = round(np.sum(-x_train[i]*np.log(x_train[i]))/math.log(100),1)
        entropy_trains[int(entropy*10)] += 1
    entropy_trains = [e/x_train.shape[0] for e in entropy_trains]
    for i in range(x_valid.shape[0]):
        entropy = round(np.sum(-x_valid[i]*np.log(x_valid[i]))/math.log(100),1)
        entropy_valid[int(entropy*10)] += 1
    entropy_valid = [e/x_valid.shape[0] for e in entropy_valid]

    xx = [i/15 for i in range(10)]
    plt.plot(xx, entropy_valid,label = "non-member",marker="o")
    print(entropy_valid)
    plt.plot(xx, entropy_trains,label = "member",marker="o")
    print(entropy_trains)

'''
x = np.load("../result/CNN/PURCHASE_None/x.npy")
y = np.load("../result/CNN/PURCHASE_None/y.npy")
c = np.load("../result/CNN/PURCHASE_None/c.npy")
PP(x,y,c,"no defense")
x = np.load("../result/CNN/PURCHASE_OUR_0.003/x.npy")
y = np.load("../result/CNN/PURCHASE_OUR_0.003/y.npy")
c = np.load("../result/CNN/PURCHASE_OUR_0.003/c.npy")
PP(x,y,c,"DAMP")
x = np.load("../result/CNN/PURCHASE_GAN_2.0/x.npy")
y = np.load("../result/CNN/PURCHASE_GAN_2.0/y.npy")
c = np.load("../result/CNN/PURCHASE_GAN_2.0/c.npy")
PP(x,y,c,"Min-max game")


x = np.load("./result/CNN/TEXAS_None/x.npy")
y = np.load("./result/CNN/TEXAS_None/y.npy")
c = np.load("./result/CNN/TEXAS_None/c.npy")
PP(x,y,c,"no defense")
x = np.load("./result/CNN/TEXAS_OUR_0.01/x.npy")
y = np.load("./result/CNN/TEXAS_OUR_0.01/y.npy")
c = np.load("./result/CNN/TEXAS_OUR_0.01/c.npy")
PP(x,y,c,"DAMP")
x = np.load("./result/CNN/TEXAS_GAN_2.0/x.npy")
y = np.load("./result/CNN/TEXAS_GAN_2.0/y.npy")
c = np.load("./result/CNN/TEXAS_GAN_2.0/c.npy")
PP(x,y,c,"Min-Max game")

'''
x = np.load("./result/CNN/TEXAS_None/x.npy")
y = np.load("./result/CNN/TEXAS_None/y.npy")
c = np.load("./result/CNN/TEXAS_None/c.npy")
PP(x,y,c,"no defense")
x = np.load("./result/CNN/TEXAS_OUR_0.01/x.npy")
y = np.load("./result/CNN/TEXAS_OUR_0.01/y.npy")
c = np.load("./result/CNN/TEXAS_OUR_0.01/c.npy")
PP(x,y,c,"DAMP")

'''
x = np.load("../result/CNN/LOCATION_None/x.npy")
y = np.load("../result/CNN/LOCATION_None/y.npy")
c = np.load("../result/CNN/LOCATION_None/c.npy")
x = np.load("./result/CNN/LOCATION_OUR_0.005/x.npy")
y = np.load("./result/CNN/LOCATION_OUR_0.005/y.npy")
c = np.load("./result/CNN/LOCATION_OUR_0.005/c.npy")
PPP(x,y,c,"no defense")

x = np.load("../result/CNN/TEXAS_OUR_0.005/x.npy")
y = np.load("../result/CNN/TEXAS_OUR_0.005/y.npy")
c = np.load("../result/CNN/TEXAS_OUR_0.005/c.npy")
PPP(x,y,c,"DAMP")
'''

plt.legend(loc='lower right')
plt.ylabel("Cumulative probability")
plt.xlabel("Generalization error")
plt.savefig("TEXAS.pdf")