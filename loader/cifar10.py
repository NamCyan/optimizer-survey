from torchvision import datasets,transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torch
import os

def load(valid_rate, batch_size):
    savefolder = "../dat/cifar10/"
    if os.path.isfile(os.path.join(savefolder, "train_{}.pth".format(batch_size))):
        train_loader = torch.load(os.path.join(savefolder,"train_{}.pth".format(batch_size)))
        valid_loader = torch.load(os.path.join(savefolder,"valid_{}.pth".format(batch_size)))
        test_loader = torch.load(os.path.join(savefolder,"test_{}.pth".format(batch_size)))
        return train_loader, valid_loader, test_loader, 10

    mean=[x/255 for x in [125.3,123.0,113.9]]
    std=[x/255 for x in [63.0,62.1,66.7]]

    trainset= datasets.CIFAR10(savefolder,train=True,download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
    test= datasets.CIFAR10(savefolder,train=False,download=True,
                                    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))

    valid_length = int(valid_rate * len(trainset))
    train_length = len(trainset) - valid_length
    train, valid = torch.utils.data.random_split(trainset, [train_length, valid_length])

    loader_train = DataLoader(train, batch_size= batch_size, sampler= RandomSampler(train))
    loader_valid = DataLoader(valid, batch_size= batch_size, sampler= SequentialSampler(valid))
    loader_test = DataLoader(test, batch_size= batch_size, sampler= SequentialSampler(test))

    torch.save(loader_train, os.path.join(savefolder, "train_{}.pth".format(batch_size)))
    torch.save(loader_valid, os.path.join(savefolder, "valid_{}.pth".format(batch_size)))
    torch.save(loader_test, os.path.join(savefolder, "test_{}.pth".format(batch_size)))
    return loader_train, loader_valid, loader_test, 10