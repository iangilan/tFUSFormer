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
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import config
from config import dir_path, model_path, upsampler, models_config, selected_model_key
from utils import iou, IoULoss
from tFUS_dataloader import train_dl, valid_dl
from time import sleep
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import EarlyStopping
from pathlib import Path

upsampler = config.upsampler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define loss functions
loss_func = nn.MSELoss()    
loss_func2 = IoULoss()
adversarial_loss = nn.BCELoss()

# Initialize the selected model from config.py
model_class = models_config[selected_model_key]['class']
model_params = models_config[selected_model_key]['params']
model = model_class(**model_params).to(device)


def train_GAN(model, data_dl):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    running_d_loss = 0.0

    for _, data in enumerate(tqdm(data_dl)):
        P = data[0].to(device)  # LR images as input to generator
        label = data[5].to(device)  # HR images as real samples for discriminator

        # Zero grad the optimizers
        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # Generate fake images
        fake_outputs = model.generate(P)

        # Train discriminator on real images
        d_real = model.discriminate(label).squeeze()
        loss_d_real = adversarial_loss(d_real, torch.ones_like(d_real))

        # Train discriminator on fake images
        d_fake = model.discriminate(fake_outputs.detach()).squeeze()
        loss_d_fake = adversarial_loss(d_fake, torch.zeros_like(d_fake))

        # Combined discriminator loss
        d_loss = (loss_d_real + loss_d_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Train generator
        g_loss = loss_func(fake_outputs, label)  # Content loss (e.g., MSE)
        fake_pred = model.discriminate(fake_outputs).squeeze()
        adversarial_g_loss = adversarial_loss(fake_pred, torch.ones_like(fake_pred))  # Adversarial loss
        total_g_loss = config.alpha * g_loss + (1 - config.alpha) * adversarial_g_loss
        total_g_loss.backward()
        optimizer_G.step()

        running_loss += g_loss.item()
        running_d_loss += d_loss.item()
        batch_iou = iou(label, fake_outputs)
        running_iou += batch_iou

    final_loss = running_loss / len(data_dl)
    final_d_loss = running_d_loss / len(data_dl)
    final_iou = running_iou / len(data_dl)

    return final_loss, final_iou

def validate_GAN(model, data_dl, epoch):
    model.eval()
    running_g_loss = 0.0
    running_d_loss_real = 0.0
    running_d_loss_fake = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for _, data in enumerate(data_dl):
            P = data[0].to(device)  # LR images as input to generator
            label = data[5].to(device)  # HR images as real samples for discriminator

            # Generate fake images
            fake_outputs = model.generate(P)

            # Discriminator loss on real images
            d_real = model.discriminate(label).squeeze()
            loss_d_real = adversarial_loss(d_real, torch.ones_like(d_real))

            # Discriminator loss on fake images
            d_fake = model.discriminate(fake_outputs).squeeze()
            loss_d_fake = adversarial_loss(d_fake, torch.zeros_like(d_fake))

            # Generator loss
            g_loss = loss_func(fake_outputs, label)

            running_g_loss += g_loss.item()
            running_d_loss_real += loss_d_real.item()
            running_d_loss_fake += loss_d_fake.item()
            batch_iou = iou(label, fake_outputs)
            running_iou += batch_iou            

    final_g_loss = running_g_loss / len(data_dl)
    final_d_loss_real = running_d_loss_real / len(data_dl)
    final_d_loss_fake = running_d_loss_fake / len(data_dl)
    final_iou = running_iou / len(data_dl)

    return final_g_loss, final_iou


def train(model, data_dl):
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for ba, data in enumerate(tqdm(data_dl)):
        P  = data[0].to(device) #LR
        label = data[5].to(device) #HR

        optimizer.zero_grad()
        
        # Check model class to determine input format
        if model.__class__.__name__ == 'tFUSFormer_5ch':
            S  = data[1].to(device)  # LR
            Vx = data[2].to(device)  # LR
            Vy = data[3].to(device)  # LR
            Vz = data[4].to(device)  # LR
            outputs = model(P, S, Vx, Vy, Vz)  # SR
        else:
            outputs = model(P)  # SR
            
        loss = loss_func(outputs, label)
        loss2 = loss_func2(outputs, label)
        loss = config.alpha*loss + (1.0-config.alpha)*loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_iou = iou(label, outputs)
        running_iou += batch_iou
        
    final_loss = running_loss / len(data_dl)  # Average loss per batch
    final_iou = running_iou / len(data_dl)  # Average IOU per batch

    return final_loss, final_iou

# validation
def validate(model, data_dl, epoch):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    with torch.no_grad():
        for ba, data in enumerate(data_dl):
            P  = data[0].to(device)
            label = data[5].to(device)

            # Check model class to determine input format
            if model.__class__.__name__ == 'tFUSFormer_5ch':
                S  = data[1].to(device)  # LR
                Vx = data[2].to(device)  # LR
                Vy = data[3].to(device)  # LR
                Vz = data[4].to(device)  # LR
                outputs = model(P, S, Vx, Vy, Vz)  # SR
            else:
                outputs = model(P)  # SR
                        
            loss = loss_func(outputs, label)
       	    loss2 = loss_func2(outputs, label)
            loss = config.alpha*loss + (1.0-config.alpha)*loss2

            running_loss += loss.item()
            batch_iou = iou(label,outputs)
            running_iou += batch_iou

    final_loss = running_loss / len(data_dl)  # Average loss per batch
    final_iou = running_iou / len(data_dl)  # Average IOU per batch

    return final_loss, final_iou

num_epochs = config.num_epochs

# Define optimizers
if model.__class__.__name__ == 'SRGAN_1ch':
    optimizer_G = optim.Adam(model.generator.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=0.0002)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
else:
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


# train
train_loss, val_loss = [], []
train_iou, val_iou   = [], []
start = time.time()

early_stopping = EarlyStopping(patience = 40, verbose = True)
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    start1 = time.time()
    if model.__class__.__name__ == 'SRGAN_1ch':
        train_epoch_loss, train_epoch_iou = train_GAN(model, train_dl)
        val_epoch_loss, val_epoch_iou     = validate_GAN(model, valid_dl, epoch)        
    else:
        train_epoch_loss, train_epoch_iou = train(model, train_dl)
        val_epoch_loss, val_epoch_iou     = validate(model, valid_dl, epoch)

    scheduler.step(val_epoch_loss)
    early_stopping(val_epoch_loss, model)   

    train_loss.append(train_epoch_loss)
    train_iou.append(train_epoch_iou)
    val_loss.append(val_epoch_loss)
    val_iou.append(val_epoch_iou)

    end = time.time()
    #print(f'Train IoU: {train_epoch_iou:.4f}, Train Loss: {train_epoch_loss:.5f}, Val IoU: {val_epoch_iou:.4f}, Val Loss: {val_epoch_loss:.4f}, Time: {end-start1:.2f} sec, Total Time: {end-start:.2f} sec')
    print(f'Train IoU: {train_epoch_iou:.4f}, Train Loss: {train_epoch_loss:.5f},\n'
      f'Valid IoU: {val_epoch_iou:.4f}, Valid Loss: {val_epoch_loss:.5f}, '
      f'Time: {end-start1:.2f} sec, Total Time: {end-start:.2f} sec')
    if early_stopping.early_stop:
        print("Early stopping")
        break

# Check if the directory does not exist
if not os.path.exists(model_path):
    # Create the directory
    os.makedirs(model_path)
    print(f"Directory '{model_path}' was created.")
else:
    print(f"Directory '{model_path}' already exists.")

full_class_name = str(model.__class__)
class_path = full_class_name.split("'")[1]  # Splits on ' and takes the second element which is the class path
model_name = f"{class_path}.pth".replace('s.', '_')
print(model_name)

torch.save(model.state_dict(), f'{model_path}/{model_name}')    
