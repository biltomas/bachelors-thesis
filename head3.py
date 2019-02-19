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

def loadData(data_path, batch):
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

def collage(data):
    if type(data) is not list:
        if data.shape[3] != 3:
            data = data.transpose(0, 2, 3, 1)
            
        images = [img for img in data]
    else:
        images = list(data)

    side = int(np.ceil(len(images)**0.5))
    for i in range(side**2 - len(images)):
        images.append(images[-1])
    collage = [np.concatenate(images[i::side], axis=0)
               for i in range(side)]
    collage = np.concatenate(collage, axis=1)
    #collage -= collage.min()
    #collage = collage / np.absolute(collage).max() * 256
    return collage

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

class VGGNet(nn.Module):
    def __init__(self, base_channels=8, num_classes=10):
        super(VGGNet, self).__init__()
        
        self.layer1 = VGGBlock(input_channels=3, 
                               layer_count=2, filter_count=base_channels*1)
        
        self.layer2 = VGGBlock(input_channels=base_channels*1, 
                               layer_count=2, filter_count=base_channels*2)
        
        self.layer3 = VGGBlock(input_channels=base_channels*2, 
                               layer_count=2, filter_count=base_channels*4)
        
        self.fc1 = nn.Linear(4*4*base_channels*4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        # print('in: ')
        # print(4*4*base_channels*4)
        
    def forward(self, x):
        out = self.layer1(x)
        # print("second layer")
        out = self.layer2(out)
        # print("third layer")
        out = self.layer3(out)
        # print("out layer")
        out = out.reshape(out.size(0), -1)
        # print('out: ')
        # print(out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        return out

torch.set_default_tensor_type('torch.cuda.FloatTensor')
print(torch.cuda.is_available())
print(torch.__version__)
# trnData, tstData, trnLabels, tstLabels = readCIFAR()
data_path = 'D:\learning_data\\train'
test_path = 'D:\learning_data\\test'
# trnData, tstData, trnLabels = loadData()
# data_iter = iter(loadData(data_path, 49))
# test_iter = iter(loadData(test_path, 32))

# trnData = torch.tensor
# trnLabels = torch.tensor
# tstData = torch.tensor
# tstLabels = torch.tensor

for batch_idx, (data, target) in enumerate(loadData(data_path, 30)):
    if batch_idx == 0:
        trnData = data
        trnLabels = target
    else:
        trnData = torch.cat((trnData, data), 0)
        trnLabels = torch.cat((trnLabels, target), 0)
    # print(batch_idx)
    # print(trnData.size())

for batch_idx, (data, target) in enumerate(loadData(test_path, 24)):
    if batch_idx == 0:
        tstData = data
        tstLabels = target
    else:
        tstData = torch.cat((tstData, data), 0)
        tstLabels = torch.cat((tstLabels, target), 0)
    # print(batch_idx)
    # print(trnData.size())

# trnData = data[0]
# trnLabels = data[1]
# tstData = test[0]
# tstLabels = test[1]
# trnData = trnData.astype(np.float32) / 255.0 - 0.5
# tstData = tstData.astype(np.float32) / 255.0 - 0.5


print('trnData')
print(trnData.size())
print(trnData.size())
print('tstData')
print(tstData.size())
print(tstData.size())
print('trnLabels')
print(trnLabels.size())
print(trnLabels.size())


# plt.subplot(1, 2, 1)
# img = collage(trnData[:16])
# print(img.shape)
# plt.imshow(img)
# plt.subplot(1, 2, 2)
# img = collage(tstData[:16])
# plt.imshow(img)
# plt.show()
trnData = trnData.numpy()
tstData = tstData.numpy()
trnLabels = trnLabels.numpy()
tstLabels = tstLabels.numpy()


trnData = trnData.astype(np.float32) / 255.0 - 0.5
tstData = tstData.astype(np.float32) / 255.0 - 0.5

model = VGGNet(base_channels=8, num_classes=int(trnLabels.max()+1))
model = model.cuda()

batch_size =  30
view_step = 1

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_acc = 0
accuracy_acc = 0
curve = []
for i in range(200):
  batch_ids = np.random.choice(trnLabels.shape[0], batch_size)
#   print(batch_ids)
#   batch_data = torch.from_numpy(trnData[batch_ids].transpose(0, 3, 1, 2))
  batch_data = torch.from_numpy(trnData[batch_ids])
#   print(batch_data.shape)
#   batch_data = torch.from_numpy(trnData[batch_ids])
  batch_labels = torch.from_numpy(trnLabels[batch_ids])
  batch_data = batch_data.cuda()
  batch_labels = batch_labels.cuda()
#   print(batch_data.is_cuda)
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


test_batch_size = 24
tstLabels_split = np.split(tstLabels, 12)
tstData_split = np.split(tstData, 12)

# print((tstLabels_split[0]))

for i in range(len(tstData_split)):
    temp_acc = 0.0
    batch_data = torch.from_numpy(tstData_split[i])
    batch_labels = torch.from_numpy(tstLabels_split[i])
    batch_data = batch_data.cuda()
    batch_labels = batch_labels.cuda()
    outputs = model(batch_data)
    # print(outputs)
    max_scores, pred_labels = torch.max(outputs, 1)
    # print(pred_labels)

    accuracy_acc += torch.sum(pred_labels == batch_labels).item() / float(test_batch_size)
    temp_acc += torch.sum(pred_labels == batch_labels).item() / float(test_batch_size)
    print ('Test batch Accuracy: {:.4f}' .format(temp_acc))
    # total_acc += accuracy_acc

accuracy_acc = accuracy_acc/len(tstData_split)
print ('Test Accuracy: {:.4f}' .format(accuracy_acc))
plt.plot(curve)
plt.show()