import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
matplotlib.use('agg')

x_1 = np.load("result/CNN/TEXAS_None_0.0/x.npy")
y_1 = np.load("result/CNN/TEXAS_None_0.0/y.npy")
x_2 = np.load("result/CNN/TEXAS_None_1.0/x.npy")
y_2 = np.load("result/CNN/TEXAS_None_1.0/y.npy")

x_1 = nn.Softmax(dim=1)(torch.from_numpy(x_1)).numpy()
x_2 = nn.Softmax(dim=1)(torch.from_numpy(x_2)).numpy()

index = np.where(y_1 == 1)[0]
x_1 = x_1[index]

index = np.where(y_2 == 1)[0]
x_2 = x_2[index]

mse = 0
max = 0
min = 10
x = [i for i in range(150)]
y = [0 for i in range(150)]
z = [0 for i in range(150)]
for i in range(x_1.shape[0]):
    t = np.linalg.norm(x_1[i] - x_2[i])
    y[int(t*100)] += 1
    for j in range(int(t*100),150):
        z[j] += 1
    mse += t
    max =t if max < t else max
    min = t if min > t else min

mse/=x_1.shape[0]
print(min,mse,max)

print(z)
z = [i/x_1.shape[0] for i in z]
#plt.plot(x,y)
plt.plot(x,z)
print(z)
plt.savefig("None.png")