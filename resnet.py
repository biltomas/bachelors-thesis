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

trnData, trnLabels, tstData, tstLabels = loadData()

model = models.resnet34(pretrained=True).cuda()
batch_size =  30
view_step = 1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

loss_acc = 0
accuracy_acc = 0
curve = []
curve1 = []
for i in range(1200):
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



plt.plot(curve)
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