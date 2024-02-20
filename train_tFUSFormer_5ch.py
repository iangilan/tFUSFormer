import scipy.io
import torch
import unfoldNd
import os
from functools import reduce, lru_cache
from einops import rearrange
from operator import mul
import matplotlib
import matplotlib.pyplot as plt
import warnings
import time
import h5py
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.optim as optim
import numpy as np
import math
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from config import dir_path, model_path
from utils import iou
from train_dataloader import tFUSFormer5chDataset, DataLoader, train_ds, valid_ds, train_dl, valid_dl

from models import tFUSFormer_5ch
# import EarlyStopping
#from pytorchtools import EarlyStopping

from time import sleep
#import collections.abc
#from itertools import repeat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_func = nn.MSELoss()    

#model = tFUSFormer_5ch(upsampler='pixelshuffle').to(device)
model = tFUSFormer_5ch(upsampler='pixelshuffledirect').to(device)
#model = tFUSFormer_5ch(upsampler='nearest+conv').to(device)

def train(model, data_dl):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for ba, data in enumerate(tqdm(data_dl)):
        P  = data[0].to(device) #LR
        S  = data[1].to(device) #LR
        Vx = data[2].to(device) #LR
        Vy = data[3].to(device) #LR
        Vz = data[4].to(device) #LR
        label = data[5].to(device) #HR

        optimizer.zero_grad()
        outputs = model(P,S,Vx,Vy,Vz) #SR
        #print('output size ',outputs.size()) # Why 4 1 50 50 50 ???????
        #print('HR size ',label.size())   #     4 1 100 100 100
        #print(torch.max(outputs))
        loss = loss_func(outputs, label)
        #loss2 = loss_func2(outputs, label)
        #loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_iou = iou(label, outputs)
        running_iou += batch_iou
        
    final_loss = running_loss / len(data_dl)  # Average loss per batch
    final_iou = running_iou / len(data_dl)  # Average IOU per batch
    #final_loss = running_loss / len(data_dl.dataset)
    #final_iou = running_iou / int(len(train_ds)/data_dl.batch_size)

    return final_loss, final_iou

# validation
def validate(model, data_dl, epoch):

    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for ba, data in enumerate(data_dl):
            P  = data[0].to(device)
            S  = data[1].to(device)
            Vx = data[2].to(device)
            Vy = data[3].to(device)
            Vz = data[4].to(device)
            label = data[5].to(device)

            outputs = model(P,S,Vx,Vy,Vz)
            loss = loss_func(outputs, label)
            #loss2 = loss_func2(outputs, label)
            #loss = loss1 + loss2

            running_loss += loss.item()
            batch_iou = iou(label,outputs)
            running_iou += batch_iou

    final_loss = running_loss / len(data_dl)  # Average loss per batch
    final_iou = running_iou / len(data_dl)  # Average IOU per batch
    #final_loss = running_loss / len(data_dl.dataset)
    #final_iou = running_iou / int(len(data_dl)/data_dl.batch_size)

    return final_loss, final_iou

optimizer = optim.Adam(model.parameters(), lr=0.0002)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 2 #100 # 51 done

# train
train_loss, val_loss = [], []
train_iou, val_iou   = [], []
start = time.time()
#early_stopping = EarlyStopping(patience = patience, verbose = True)
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    start1 = time.time()
    train_epoch_loss, train_epoch_iou = train(model, train_dl)
    val_epoch_loss, val_epoch_iou     = validate(model, valid_dl, epoch)

    train_loss.append(train_epoch_loss)
    train_iou.append(train_epoch_iou)
    val_loss.append(val_epoch_loss)
    val_iou.append(val_epoch_iou)

    end = time.time()
    print(f'Train IoU: {train_epoch_iou:.4f}, Train Loss: {train_epoch_loss:.5f}, Val IoU: {val_epoch_iou:.4f}, Time: {end-start1:.2f} sec, Total Time: {end-start:.2f} sec')


# Check if the directory does not exist
if not os.path.exists(model_path):
    # Create the directory
    os.makedirs(model_path)
    print(f"Directory '{model_path}' was created.")
else:
    print(f"Directory '{model_path}' already exists.")

torch.save(model.state_dict(), f'{model_path}/tFUSFormer_5ch_model.pth')    
