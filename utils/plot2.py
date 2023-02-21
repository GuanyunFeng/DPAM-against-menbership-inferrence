import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import math
matplotlib.use('agg')

total_width, n = 0.15, 3
width = total_width / n

def PP(a, name):
    c = [0 for i in range(11)]
    for i in range(a.shape[0]):
        aa = a[i] if a[i]>0.5 else 1-a[i]
        c[int(aa*10)] += 1
    print(c)
    if name =="no defense":
        x = [i/10 for i in range(len(c))]
    elif name == "defense gan":
        x = [i/10+width for i in range(len(c))]
    elif name == "defense our":
        x = [i/10+width for i in range(len(c))]
    plt.bar(x,c,width = width,label=name)


for model_name in ["knn"]:
    print(model_name)
    a1 = np.load("{}_none.npy".format(model_name))
    PP(a1,"no defense")
    a2 = np.load("{}_gan.npy".format(model_name))
    PP(a2,"defense gan")
    a3 = np.load("{}_our.npy".format(model_name))
    PP(a3,"defense our")

plt.legend()
plt.savefig("1.png")