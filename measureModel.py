import torch
import torch.nn.functional as F

from loadData import loadData
from test import test

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

# tento skript nacte jiz ulozeny model a vyhodnoti meriky pro jednotlive tridy
# pro nacitani je nutne mit k dispozici tridy, ktere puvodne slouzily k vytvoreni modelu

class lin(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lin, self).__init__()

        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class lin2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lin2, self).__init__()

        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x1, x2):
        x2=x2.view(x2.size()[0],-1)
        x = torch.cat([x2, x1], dim=1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class VGGBlock(nn.Module):
    def __init__(self, input_channels, layer_count, filter_count):
      super(VGGBlock, self).__init__()
      layers = []
      for i in range(layer_count):
        layers.append(nn.Conv2d(input_channels, filter_count, kernel_size=3, stride=1, padding=1))
        input_channels = filter_count
        layers.append(nn.BatchNorm2d(filter_count))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2))

      self.block = nn.Sequential(*layers)
      
    def forward(self, x):
      out = self.block(x)
      return out      

class VGGNet(nn.Module):
    def __init__(self, base_channels=8, num_classes=6):
        super(VGGNet, self).__init__()
        
        self.layer1 = VGGBlock(input_channels=3, 
                               layer_count=2, filter_count=base_channels*1)
        
        self.layer2 = VGGBlock(input_channels=base_channels*1, 
                               layer_count=2, filter_count=base_channels*2)
        
        self.layer3 = VGGBlock(input_channels=base_channels*2, 
                               layer_count=2, filter_count=base_channels*4)
        
        self.layer4 = VGGBlock(input_channels=base_channels*4, 
                               layer_count=2, filter_count=base_channels*8)
        
        self.fc1 = nn.Linear(4*2*base_channels, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        base_channels=8
        self.down1 = VGGBlock(input_channels=3, 
                               layer_count=2, filter_count=base_channels*1)
        
        self.down2 = VGGBlock(input_channels=base_channels*1, 
                               layer_count=2, filter_count=base_channels*2)
        
        self.down3 = VGGBlock(input_channels=base_channels*2, 
                               layer_count=2, filter_count=base_channels*4)
        
        self.down4 = VGGBlock(input_channels=base_channels*4, 
                               layer_count=2, filter_count=base_channels*8)
        self.fc1 = lin(1536, 128)
        self.fc2 = lin2(4224, 32)
        self.fc3 = lin2(32800, n_classes)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.down4(x3)
        x = self.fc1(x, x3)
        x = self.fc2(x, x2)
        x = self.fc3(x, x1)
        return F.sigmoid(x)

# tato funkce nameri metriky pro jednotlive tridy

def sortAndMeasure(model, tstLabels, tstData):
    tstLabels_split = np.split(tstLabels, 1)
    tstData_split = np.split(tstData, 1)
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

    for i in range(6):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        shot = 0
        for x in range(test_batch_size):
            if tstLabels_split[0][x] == i:
                shot += 1
                if pred_labels[x] == i:
                    TP +=1
                else:
                    FN += 1
                    print('Expected {} got {}' .format(i, pred_labels[x]))
            else:
                if pred_labels[x] == i:
                    FP +=1
                else:
                    TN += 1
        accuracy = (TP + TN)/(TP + TN + FP + FN)
        if (TP + FP) != 0:
            precision = TP/(TP + FP)
        else:
            precision = 0
        recall = TP/(TP + FN)
        if (precision+recall) != 0:
            f1 = 2*(precision*recall)/(precision+recall)
        else:
            f1 = 0
        print ('Category {} Accuracy: {:.4f} Precision: {:.4f} Recall: {:.4f} F-measure: {:.4f}' .format(i, accuracy, precision, recall, f1))
    accuracy_acc = accuracy_acc/len(tstData_split)
    print ('Test Accuracy: {:.4f}' .format(accuracy_acc))

# nacteni dat
trnData, trnLabels, tstData, tstLabels = loadData()
# nacteni modelu
model = torch.load( r'C:\\Users\\tomas\\Documents\\bp\\bp2018\\model\\resnet-90.pth').cuda()

sortAndMeasure(model, tstLabels, tstData)