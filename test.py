import torch
import torch.nn.functional as F

from loadData import loadData

from torch import nn
from torch import utils
import pickle
import os
import torchvision
from os import path
accelerator = 'cu90' if path.exists('/opt/bin/nvidia-smi') else 'cpu'

import matplotlib.pyplot as plt

import numpy as np

import sys
from torchvision import models
import datetime

# funkce změří přesnost dodaneho modelu na dodanych testovacich datech a v pripade odkomentovani model ulozi
def test(tstLabels, tstData, model):
    tstLabels_split = np.split(tstLabels, 3)
    tstData_split = np.split(tstData, 3)
    accuracy_acc = 0
    test_batch_size = len(tstLabels_split[0])

    for i in range(len(tstData_split)):
        temp_acc = 0.0
        batch_data = torch.from_numpy(tstData_split[i])
        batch_labels = torch.from_numpy(tstLabels_split[i])
        batch_data = batch_data.cuda()
        batch_labels = batch_labels.cuda()
        outputs = model(batch_data)
        max_scores, pred_labels = torch.max(outputs, 1)
        accuracy_acc += torch.sum(pred_labels == batch_labels).item() / float(test_batch_size)
        temp_acc += torch.sum(pred_labels == batch_labels).item() / float(test_batch_size)

    accuracy_acc = accuracy_acc/len(tstData_split)
    print ('Test Accuracy: {:.4f}' .format(accuracy_acc))
    time = datetime.datetime.now()
    path = r'C:\\Users\\tomas\\Documents\\bp\\bp2018\\model\\' + time.strftime("%m-%d-%Y-%H-%M-%S") + '.pth'
    # torch.save(model, path)
    return accuracy_acc