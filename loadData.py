from torch import nn
from torch import utils
import pickle
import os
import torchvision
import torch
from os import path
accelerator = 'cu90' if path.exists('/opt/bin/nvidia-smi') else 'cpu'


# %matplotlib inline  
import matplotlib.pyplot as plt

import numpy as np

import sys
from skimage.transform import resize

def readData(data_path, batch):
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    # print(train_dataset)
    train_loader = utils.data.DataLoader(
        train_dataset,
        batch_size=batch,
        num_workers=0,
        shuffle=True,
        drop_last=True
    )
    # print(train_loader)
    return train_loader

def loadData ():
    data_path = 'D:\learning_data2\\train'
    test_path = 'D:\learning_data2\\test'

    test_batch_size = 10
    train_batch_size = 30

    for batch_idx, (data, target) in enumerate(readData(data_path, train_batch_size)):
        if batch_idx == 0:
            trnData = data
            trnLabels = target
        else:
            trnData = torch.cat((trnData, data), 0)
            trnLabels = torch.cat((trnLabels, target), 0)
        # print(batch_idx)
        # print(trnData.size())

    for batch_idx, (data, target) in enumerate(readData(test_path, test_batch_size)):
        if batch_idx == 0:
            tstData = data
            tstLabels = target
        else:
            tstData = torch.cat((tstData, data), 0)
            tstLabels = torch.cat((tstLabels, target), 0)

    trnData = trnData.numpy()
    tstData = tstData.numpy()
    trnLabels = trnLabels.numpy()
    tstLabels = tstLabels.numpy()

    trnData = trnData.astype(np.float32) / 255.0 - 0.5
    tstData = tstData.astype(np.float32) / 255.0 - 0.5

    return (trnData, trnLabels, tstData, tstLabels)
