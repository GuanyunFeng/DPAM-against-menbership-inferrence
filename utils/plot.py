import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
matplotlib.use('agg')

x = np.load("../result/CNN/LOCATION_OUR_0.005/x.npy")
y = np.load("../result/CNN/LOCATION_OUR_0.005/y.npy")
c = np.load("../result/CNN/LOCATION_OUR_0.005/c.npy")

x_last = x[-1,:,:]
y_last = y[-1,:,:]
c_last = c[-1,:]
x_last = nn.Softmax(dim=1)(torch.from_numpy(x_last)).numpy()

class_index = 15
index = np.where(y_last == 0)[0]
x_last_valid = x_last[index]
c_last_valid = c_last[index]
index = np.where(y_last == 1)[0]
x_last_train = x_last[index]
c_last_train = c_last[index]

index = np.where(c_last_train== class_index)[0]
x_last0 = x_last_train[index]
xx = [i for i in range(30)]
for i in range(int(x_last0.shape[0])):
    plt.plot(xx,x_last0[i])

plt.title("LOCATION, with defense")
plt.ylabel("Prediction Probability")
plt.xlabel("class label")
plt.savefig("15_d_train.pdf")
plt.close()

index = np.where(c_last_valid== class_index)[0]
x_last0 = x_last_valid[index]
xx = [i for i in range(30)]
for i in range(int(x_last0.shape[0])):
    plt.plot(xx,x_last0[i])

plt.title("LOCATION, with defense")
plt.ylabel("Prediction Probability")
plt.xlabel("class label")
plt.savefig("15_d_val.pdf")
plt.close()

x = np.load("../result/CNN/LOCATION_None/x.npy")
y = np.load("../result/CNN/LOCATION_None/y.npy")
c = np.load("../result/CNN/LOCATION_None/c.npy")

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

index = np.where(c_last_train== class_index)[0]
x_last0 = x_last_train[index]
xx = [i for i in range(30)]
for i in range(int(x_last0.shape[0])):
    plt.plot(xx,x_last0[i])

plt.title("LOCATION, without defense")
plt.ylabel("Prediction Probability")
plt.xlabel("class label")
plt.savefig("15_train.pdf")
plt.close()

index = np.where(c_last_valid== class_index)[0]
x_last0 = x_last_valid[index]
xx = [i for i in range(30)]
for i in range(int(x_last0.shape[0])):
    plt.plot(xx,x_last0[i])

plt.title("LOCATION, without defense")
plt.ylabel("Prediction Probability")
plt.xlabel("class label")
plt.savefig("15_val.pdf")
plt.close()