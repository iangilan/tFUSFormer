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
import config
from config import dir_path, model_path, upsampler
from utils import iou
from dataloader_tFUSFormer5ch import tFUSFormer5chDataset, DataLoader, train_ds, valid_ds, test_ds, train_dl, valid_dl, test_dl#, unforeseen_test_ds, unforeseen_test_dl
from dataloader_tFUSFormer5ch import scaler_Phigh, scaler_Plow, scaler_Slow, scaler_Vxlow, scaler_Vylow, scaler_Vzlow
from models import tFUSFormer_5ch
from time import sleep

upsampler = config.upsampler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = tFUSFormer_5ch(upsampler=upsampler).to(device)
model.load_state_dict(torch.load(f'{model_path}/tFUSFormer_5ch_model.pth'))
model.eval()

LR_list, HR_list, SR_list = [], [], []
with torch.no_grad():
    for lr, sk_lr, Vx_lr, Vy_lr, Vz_lr, hr in test_dl:
        LR_list.append(lr.squeeze(1))
        HR_list.append(hr.squeeze(1))
        SR_list.append(model(lr.to(device),sk_lr.to(device), Vx_lr.to(device), Vy_lr.to(device), Vz_lr.to(device)).squeeze(1))

# Convert lists of tensors into single tensors
LR = torch.cat(LR_list, dim=0)  # This will be of shape [18, 4, 25, 25, 25]
HR = torch.cat(HR_list, dim=0)
SR = torch.cat(SR_list, dim=0)

N_test  = LR.shape[0]
nx_low  = LR.shape[1]
ny_low  = LR.shape[2]
nz_low  = LR.shape[3]
nx_high = HR.shape[1]
ny_high = HR.shape[2]
nz_high = HR.shape[3]
nxyz_high = nx_high*ny_high*nz_high
nxyz_low  = nx_low*ny_low*nz_low
###################################################
# rescaling SR
###################################################
SS=torch.zeros((N_test,nx_high,ny_high,nz_high))
LL=torch.zeros((N_test,nx_low,ny_low,nz_low))
HH=torch.zeros((N_test,nx_high,ny_high,nz_high))


for i in range(N_test):
    SS[i,:,:,:] = SR[i,:,:,:]
    LL[i,:,:,:] = LR[i,:,:,:]
    HH[i,:,:,:] = HR[i,:,:,:]

SS = SS.reshape(N_test,nxyz_high)
LL = LL.reshape(N_test,nxyz_low)
HH = HH.reshape(N_test,nxyz_high)
SS = scaler_Phigh.inverse_transform(SS)
HH = scaler_Phigh.inverse_transform(HH)
LL = scaler_Plow.inverse_transform(LL)
rescaled_SR = SS.reshape(N_test,nx_high,ny_high,nz_high)
rescaled_LR = LL.reshape(N_test,nx_low,ny_low,nz_low)
rescaled_HR = HH.reshape(N_test,nx_high,ny_high,nz_high)
###################################################
IoU_vec = np.zeros((N_test))
dist_vec = np.zeros((N_test))
for sample in range(N_test):
    def iou_individual(targets, inputs): 
        smooth = 1.0e-6
        tmp_inputs  = np.zeros(inputs.shape)
        tmp_targets = np.zeros(targets.shape)
        #========================
        # FWHM
        #========================
        tmp_inputs[inputs>=0.5*np.max(inputs)]   = 1
        tmp_targets[targets>=0.5*np.max(inputs)] = 1
 
        intersection = (tmp_inputs.flatten() * tmp_targets.flatten()).sum()
        total = (tmp_inputs.flatten() + tmp_targets.flatten()).sum()
        union = total - intersection
        IoU = (intersection + smooth)/(union + smooth)
        return IoU
    
    def dist_individual(targets, inputs): 
        #=========================================
        # indices for argmax of inputs and targets
        #=========================================
        ind1 = np.unravel_index(np.argmax(inputs), inputs.shape)
        ind2 = np.unravel_index(np.argmax(targets), targets.shape)
        ind1 = np.array([ind1[0], ind1[1], ind1[2]])
        ind2 = np.array([ind2[0], ind2[1], ind2[2]])
        #print('argmax of inputs',ind1)
        dist = np.linalg.norm(ind1-ind2)/2
        return dist

    LR_slice = rescaled_LR[sample,:,12,:]
    SR_slice = rescaled_SR[sample,:,50,:]
    HR_slice = rescaled_HR[sample,:,50,:]

    plt.figure(figsize=(22,22))


    folder_path = 'test_results/4x_5ch_foreseen/plot/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    iou_individual = iou_individual(rescaled_HR[sample,:,:,:], rescaled_SR[sample,:,:,:])
    IoU_vec[sample] = iou_individual
    
    dist_individual = dist_individual(rescaled_HR[sample,:,:,:], rescaled_SR[sample,:,:,:])
    dist_vec[sample] = dist_individual

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 6.5))
    fig.suptitle('IoU = %f' %iou_individual,fontsize=25)
    colorscheme = 'hot' # jet, seismic, turbo

    axs[0].imshow(LR_slice, cmap=colorscheme)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(SR_slice, cmap=colorscheme)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(HR_slice, cmap=colorscheme)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    # axs[0].set_title(f'LR max = {np.max(rescaled_LR[sample,:,:,:]):.4f}', fontsize=20)
    # axs[1].set_title(f'SR max = {np.max(rescaled_SR[sample,:,:,:]):.4f}', fontsize=20)
    # axs[2].set_title(f'HR max = {np.max(rescaled_HR[sample,:,:,:]):.4f}', fontsize=20)
    plt.subplots_adjust(wspace = 0, hspace = 0)

    # Set the file path with the folder path variable included
    file_path2 = os.path.join(folder_path, '5ch_superformer_4x_sample%d.jpg' %sample)
    # Save the figure with the specified file path
    plt.savefig(file_path2, dpi=600, bbox_inches='tight', pad_inches=0)
iou_mean = np.mean(IoU_vec)
iou_median = np.median(IoU_vec)
dist_mean = np.mean(dist_vec)
dist_median = np.median(dist_vec)
print('=================================================')
print('IoU_mean for N_test cases   = ',iou_mean)
print('IoU_median for N_test cases = ',iou_median)
print('=================================================')   
print('dist_mean for N_test cases   = ',dist_mean)
print('dist_median for N_test cases = ',dist_median)
print('=================================================') 

#=========================================
# Create the folder if it does not exist
folder_path1 = 'test_results/4x_5ch_foreseen/'
if not os.path.exists(folder_path1):
    os.makedirs(folder_path1)

file_path1 = os.path.join(folder_path1, 'iou_results.txt')

with open(file_path1, 'w') as f:
    print('=================================================', file=f)
    print('IoU_mean for N_test cases   = ', iou_mean, file=f)
    print('IoU_median for N_test cases = ', iou_median, file=f)
    print('=================================================', file=f)
   
file_path3 = os.path.join(folder_path1, 'dist_results.txt')

with open(file_path3, 'w') as f:
    print('=================================================', file=f)
    print('dist_mean for N_test cases   = ', dist_mean, file=f)
    print('dist_median for N_test cases = ', dist_median, file=f)
    print('=================================================', file=f)    

file_path4 = os.path.join(folder_path1, 'IoU_vec.txt')
with open(file_path4, 'w') as file:
    for value in IoU_vec:
        file.write(str(value) + '\n')
        
np.save(folder_path1 + 'LR.npy', rescaled_LR)
np.save(folder_path1 + 'SR.npy', rescaled_SR)
np.save(folder_path1 + 'HR.npy', rescaled_HR)        

np.save(folder_path1 + 'LR.npy', rescaled_LR)
np.save(folder_path1 + 'SR.npy', rescaled_SR)
np.save(folder_path1 + 'HR.npy', rescaled_HR)

# Get one batch from test_dl
lr, sk_lr, Vx_lr, Vy_lr, Vz_lr, hr = next(iter(test_dl))

# Inference time check
model = model.to(device)
lr    = lr[0:1].to(device)
sk_lr = sk_lr[0:1].to(device)
Vx_lr = Vx_lr[0:1].to(device)
Vy_lr = Vy_lr[0:1].to(device)
Vz_lr = Vz_lr[0:1].to(device)

# Create CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record start event
start_event.record()

# Perform inference
with torch.no_grad():
    output = model(lr, sk_lr, Vx_lr, Vy_lr, Vz_lr).to(device)

# Record end event
end_event.record()

# Wait for events to complete and measure time
torch.cuda.synchronize()
inference_time = start_event.elapsed_time(end_event) / 1000.0  # convert from milliseconds to seconds
print(f"Inference time: {inference_time:.4f} seconds")
