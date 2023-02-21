import torch
import torchvision
from torchvision.datasets import CIFAR10, CIFAR100, MNIST
from torchvision import datasets, models, transforms
import numpy as np
from PIL import Image
import random

class custum_CIFAR10(CIFAR10):

    def __init__(self, start=0, end=2500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % (self.end - self.start)
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def load_cifar10(train_size, valid_size, attack_size, batch_size):
    data_train_target = custum_CIFAR10(0, train_size, '../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    data_test_target = custum_CIFAR10(train_size, train_size + valid_size, '../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))

    data_test_attack = custum_CIFAR10(0, attack_size, '../data', train=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))

    train_loader = torch.utils.data.DataLoader(data_train_target, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test_target, batch_size=batch_size, shuffle=True)
    attack_loader = torch.utils.data.DataLoader(data_test_attack, batch_size=batch_size, shuffle=True)
    dataloader = {"train": train_loader, "val": test_loader, "attack": attack_loader}

    return dataloader




class custum_CIFAR100(CIFAR100):

    def __init__(self, start=0, end=2500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % (self.end - self.start)
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def load_cifar100(train_size, valid_size, attack_size, batch_size):
    data_train_target = custum_CIFAR100(0, train_size, '../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
    data_test_target = custum_CIFAR100(train_size, train_size + valid_size, '../data', train=True, transform=transforms.Compose([
                       transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))

    data_test_attack = custum_CIFAR100(0, attack_size, '../data', train=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))

    train_loader = torch.utils.data.DataLoader(data_train_target, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test_target, batch_size=batch_size, shuffle=False)
    attack_loader = torch.utils.data.DataLoader(data_test_attack, batch_size=batch_size, shuffle=False)
    dataloader = {"train": train_loader, "val": test_loader, "attack": attack_loader}

    return dataloader




class Dataset_MNIST(MNIST):

    def __init__(self, start=0, end=2500, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.end = end
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % (self.end - self.start)
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def load_mnist(train_size, valid_size, attack_size, batch_size):
    data_train_target = Dataset_MNIST(0, train_size, '../data', train=True, download=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    data_test_target = Dataset_MNIST(train_size, train_size+valid_size, '../data', train=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    data_test_attack = Dataset_MNIST(0, attack_size, '../data', train=False,
                       transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    train_loader = torch.utils.data.DataLoader(data_train_target, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test_target, batch_size=batch_size, shuffle=True)
    attack_loader = torch.utils.data.DataLoader(data_test_attack, batch_size=batch_size, shuffle=True)
    dataloader = {"train": train_loader, "val": test_loader, "attack": attack_loader}
    return dataloader


class Dataset_PURCHASE():
    def __init__(self, start=0, end=2500,data_dir="../data"):
        d = np.load(data_dir+'/purchase100.npz')
        self.data = d['features']
        self.targets = d['labels']
        self.start = start
        self.end = end
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]

    def __getitem__(self, index):
        index = index % (self.end - self.start)
        img, target = self.data[index], int(self.targets[index])
        return img, target
    
    def __len__(self):
        return self.end - self.start

def load_PURCHASE(train_size, valid_size, attack_size, batch_size):
    data_train_target = Dataset_PURCHASE(0, train_size, '../data')
    data_test_target = Dataset_PURCHASE(train_size, train_size+valid_size, '../data')
    data_test_attack = Dataset_PURCHASE(train_size+valid_size, train_size+valid_size+attack_size, '../data')

    train_loader = torch.utils.data.DataLoader(data_train_target, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(data_test_target, batch_size=batch_size, shuffle=True)
    attack_loader = torch.utils.data.DataLoader(data_test_attack, batch_size=batch_size, shuffle=True)
    dataloader = {"train": train_loader, "val": test_loader, "attack": attack_loader}
    return dataloader


class Dataset_TEXAS():
    def __init__(self, start=0, end=2500,data_dir="../data"):
        #load feats
        self.data = np.load(data_dir+'/texas/feats.npy')
        self.targets = np.load(data_dir+'/texas/labels.npy')
        self.start = start
        self.end = end
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]

    def __getitem__(self, index):
        index = index % (self.end - self.start)
        img, target = self.data[index], int(self.targets[index])
        return img, target
    
    def __len__(self):
        return self.end - self.start

def load_TEXAS(train_size, valid_size, attack_size, batch_size):
    data_train_target = Dataset_TEXAS(0, train_size, '../data')
    data_test_target = Dataset_TEXAS(train_size, train_size+valid_size, '../data')
    data_test_attack = Dataset_TEXAS(train_size+valid_size, train_size+valid_size+attack_size, '../data')

    train_loader = torch.utils.data.DataLoader(data_train_target, batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test_target, batch_size=batch_size, shuffle=False)
    attack_loader = torch.utils.data.DataLoader(data_test_attack, batch_size=batch_size, shuffle=False)
    dataloader = {"train": train_loader, "val": test_loader, "attack": attack_loader}
    return dataloader


class Dataset_LOCATION():
    def __init__(self, start=0, end=2500,data_dir="../data"):
        #load feats
        self.data = np.load(data_dir+'/location/feats.npy')
        self.targets = np.load(data_dir+'/location/labels.npy')
        self.start = start
        self.end = end
        self.data = self.data[start:end]
        self.targets = self.targets[start:end]

    def __getitem__(self, index):
        index = index % (self.end - self.start)
        img, target = self.data[index], int(self.targets[index])
        return img, target
    
    def __len__(self):
        return self.end - self.start

def load_LOCATION(train_size, valid_size, attack_size, batch_size):
    data_train_target = Dataset_LOCATION(0, train_size, '../data')
    data_test_target = Dataset_LOCATION(train_size, train_size+valid_size, '../data')
    data_test_attack = Dataset_LOCATION(train_size+valid_size, train_size+valid_size+attack_size, '../data')

    train_loader = torch.utils.data.DataLoader(data_train_target, batch_size=batch_size,shuffle=False)
    test_loader = torch.utils.data.DataLoader(data_test_target, batch_size=batch_size, shuffle=False)
    attack_loader = torch.utils.data.DataLoader(data_test_attack, batch_size=batch_size, shuffle=False)
    dataloader = {"train": train_loader, "val": test_loader, "attack": attack_loader}
    return dataloader

class DPAM_loader():
    def __init__(self, dataset, batch_size=64,shuffle=False):
        self.class_num = 0
        for i in range(len(dataset)):
            _, label = dataset[i]
            if label + 1 > self.class_num:
                self.class_num = label + 1

        self.len = int(len(dataset)/batch_size)

        self.new_dataset = [[] for i in range(self.class_num)]
        for i in range(len(dataset)):
            data, label = dataset[i]
            self.new_dataset[label].append(data)
        
        self.batch_size = batch_size

    def __getitem__(self, index):

        if index >= self.__len__():
            raise StopIteration()

        half_batch = int(self.batch_size/2)
        seq = np.random.randint(self.class_num,size=half_batch)
        batch_x = [None for i in range(self.batch_size)]
        for j in range(half_batch):
            index = seq[j]
            ids = np.random.randint(len(self.new_dataset[index]),size=2)
            batch_x[j] = self.new_dataset[index][ids[0]]
            batch_x[j+half_batch] = self.new_dataset[index][ids[1]]
        
        seq_tensor = torch.from_numpy(seq)
        batch_y = torch.cat((seq_tensor,seq_tensor),0)
        batch_x = torch.tensor(batch_x)
        return batch_x,batch_y
    
    def __len__(self):
        return self.len

if __name__=="__main__":
    data_train_target = Dataset_LOCATION(0, 3000, '../data')
    loader = DPAM_loader(data_train_target,seed=1)
    print(len(loader))