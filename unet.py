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


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class lin(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lin, self).__init__()

        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 32)
        self.down1 = down(32, 64)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 256)
        self.fc1 = lin(256, 128)
        self.fc2 = lin(128, 64)
        self.fc3 = lin(64, 32)
        self.fc4 = lin(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        print('out: ')
        print(x.size())
        x = self.fc1(x, x4)
        x = self.fc2(x, x3)
        x = self.fc3(x, x2)
        x = self.fc4(x, x1)
        return F.sigmoid(x)

trnData, trnLabels, tstData, tstLabels = loadData()

model = UNet(n_channels=3, n_classes=2).cuda()
batch_size =  30
view_step = 1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_acc = 0
accuracy_acc = 0
curve = []
for i in range(300):
  batch_ids = np.random.choice(trnLabels.shape[0], batch_size)
  batch_data = torch.from_numpy(trnData[batch_ids])
  batch_labels = torch.from_numpy(trnLabels[batch_ids])

  batch_data = batch_data.cuda()
  batch_labels = batch_labels.cuda()

  outputs = model(batch_data)
  loss = criterion(outputs, batch_labels)
  
  # compute ratio of correct label predictions
  max_scores, pred_labels = torch.max(outputs, 1)
  accuracy_acc += torch.sum(pred_labels == batch_labels).item() / float(batch_size) 
        
  # Backward and optimize
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  loss_acc += loss.item()

  if i % view_step == view_step-1:
    print ('Iteration {}, Loss: {:.4f} Accuracy: {:.4f}' .format(i, loss_acc / view_step, accuracy_acc / view_step))
    curve.append(accuracy_acc / view_step)
    loss_acc = 0
    accuracy_acc = 0 