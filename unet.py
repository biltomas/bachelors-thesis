import torch
import torch.nn.functional as F

from loadData import loadData

from torch import nn
from torch import utils
from test import test

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

        # print('lin_x: ')
        # print(x.size())
        x = self.linear(x)
        return x

class lin2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(lin2, self).__init__()

        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        x2=x2.view(x2.size()[0],-1)
        x = torch.cat([x2, x1], dim=1)
        x = x.reshape(x.size(0), -1)

        # print('lin_x: ')
        # print(x.size())
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
        # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        layers.append(nn.MaxPool2d(kernel_size=2))

      self.block = nn.Sequential(*layers)
      
    def forward(self, x):
      out = self.block(x)
      return out      


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # self.inc = inconv(n_channels, 4)
        # self.down1 = down(4, 8)
        # self.down2 = down(8, 16)
        # self.down3 = down(16, 32)
        # self.down4 = down(32, 32)
        # self.down5 = down(32,32)
        # self.fc1 = lin(1024, 128)
        # self.fc2 = lin(64, 32)
        # self.fc3 = lin(16, 8)
        # self.fc4 = lin(4, n_classes)
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
        # x1 = self.inc(x)
        # print('inc: ')
        # print(x1.size())
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.down3(x3)
        # print('x4: ')
        # print(x4.size())
        # x = self.down4(x4)
        # x= self.down5(x)
        # print('out: ')
        # print(x.size())
        # x = self.fc1(x, x4)
        # x = self.fc2(x, x3)
        # x = self.fc3(x, x2)
        # x = self.fc4(x, x1)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.down4(x3)
        # print('x: ')
        # print(x.size())
        # print('x3: ')
        # print(x3.size())
        x = self.fc1(x, x3)
        # print('x: ')
        # print(x.size())
        # print('x2: ')
        # print(x2.size())
        x = self.fc2(x, x2)
        x = self.fc3(x, x1)
        return F.sigmoid(x)

trnData, trnLabels, tstData, tstLabels = loadData()

model = UNet(n_channels=3, n_classes=6).cuda()
batch_size =  30
view_step = 1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

loss_acc = 0
accuracy_acc = 0
curve = []
curve1 = []
for i in range(7000):
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
    curve.append(test(tstLabels, tstData, model) / view_step)
    curve1.append(loss_acc / view_step)
    loss_acc = 0
    accuracy_acc = 0 

plt.subplot(211)
plt.plot(curve)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.subplot(212)
plt.plot(curve1)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()