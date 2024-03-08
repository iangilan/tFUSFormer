import scipy.io
import torch
import os
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
from tqdm import tqdm, trange
import config
from config import model_path, upsampler, models_config, selected_model_key, test_data_mode
from utils import iou
from tFUS_dataloader import test_dl, unforeseen_test_dl
from models import tFUSFormer_5ch
from time import sleep
from config import dir_data
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_class = models_config[selected_model_key]['class']

model_params = models_config[selected_model_key]['params']
model = model_class(**model_params).to(device)

full_class_name = str(model.__class__)
class_path = full_class_name.split("'")[1]  # Splits on ' and takes the second element which is the class path
model_name = f"{class_path}".replace('s.', '_')
print("Model name = ",model_name)
print("Test dataset type = ", config.test_data_mode)

model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()

def load_scaler_from_hdf5(filepath, scaler_name):
    with h5py.File(filepath, 'r') as f:
        group = f[scaler_name]
        feature_range = group['feature_range'][:]
        scaler = MinMaxScaler(feature_range=(feature_range[0], feature_range[1]))
        
        # Load and set the necessary attributes
        for attribute in ['scale_', 'min_', 'data_min_', 'data_max_', 'data_range_']:
            if attribute in group:
                setattr(scaler, attribute, group[attribute][:])
        
        # Set n_features_in_ based on the shape of scale_ or min_
        if 'scale_' in group:
            scaler.n_features_in_ = group['scale_'][:].shape[0]

        # Explicitly handle n_samples_seen_ for completeness, if it was saved
        if 'n_samples_seen_' in group:
            setattr(scaler, 'n_samples_seen_', group['n_samples_seen_'][()])

    return scaler

scaler_Phigh = load_scaler_from_hdf5(f'{dir_data}/scaler_P_HR.hdf5',  'scaler_Phigh')
scaler_Plow  = load_scaler_from_hdf5(f'{dir_data}/scaler_P_LR.hdf5',  'scaler_Plow')
scaler_Slow  = load_scaler_from_hdf5(f'{dir_data}/scaler_S_LR.hdf5',  'scaler_Slow')
scaler_Vxlow = load_scaler_from_hdf5(f'{dir_data}/scaler_Vx_LR.hdf5', 'scaler_Vxlow')
scaler_Vylow = load_scaler_from_hdf5(f'{dir_data}/scaler_Vy_LR.hdf5', 'scaler_Vylow')
scaler_Vzlow = load_scaler_from_hdf5(f'{dir_data}/scaler_Vz_LR.hdf5', 'scaler_Vzlow')

#===================================================
LR_list, HR_list = [], []
SK_list, VX_list, VY_list, VZ_list = [], [], [], []
with torch.no_grad():
    for lr, sk_lr, Vx_lr, Vy_lr, Vz_lr, hr in unforeseen_test_dl if test_data_mode == 'unseen' else test_dl:
        LR_list.append(lr.squeeze(1))
        SK_list.append(sk_lr.squeeze(1))
        VX_list.append(Vx_lr.squeeze(1))
        VY_list.append(Vy_lr.squeeze(1))
        VZ_list.append(Vz_lr.squeeze(1))
        HR_list.append(hr.squeeze(1))

LR = torch.cat(LR_list, dim=0)  # This will be of shape [N_test, 25, 25, 25]
HR = torch.cat(HR_list, dim=0)
SK = torch.cat(SK_list, dim=0)
VX = torch.cat(VX_list, dim=0)
VY = torch.cat(VY_list, dim=0)
VZ = torch.cat(VZ_list, dim=0)

N_test  = LR.shape[0]
nx_low  = LR.shape[1]
ny_low  = LR.shape[2]
nz_low  = LR.shape[3]
nx_high = HR.shape[1]
ny_high = HR.shape[2]
nz_high = HR.shape[3]
nxyz_high = nx_high*ny_high*nz_high
nxyz_low  = nx_low*ny_low*nz_low

LL  = torch.zeros((N_test,nx_low,ny_low,nz_low))
HH  = torch.zeros((N_test,nx_high,ny_high,nz_high))
SSK = torch.zeros((N_test,nx_low,ny_low,nz_low))
VVX = torch.zeros((N_test,nx_low,ny_low,nz_low))
VVY = torch.zeros((N_test,nx_low,ny_low,nz_low))
VVZ = torch.zeros((N_test,nx_low,ny_low,nz_low))


for i in range(N_test):
    LL[i,:,:,:]  = LR[i,:,:,:]
    HH[i,:,:,:]  = HR[i,:,:,:]
    SSK[i,:,:,:] = SK[i,:,:,:]
    VVX[i,:,:,:] = VX[i,:,:,:]
    VVY[i,:,:,:] = VY[i,:,:,:]
    VVZ[i,:,:,:] = VZ[i,:,:,:]
    
LL  = LL.reshape(N_test,nxyz_low)
SSK = SSK.reshape(N_test,nxyz_low)
VVX = VVX.reshape(N_test,nxyz_low)
VVY = VVY.reshape(N_test,nxyz_low)
VVZ = VVZ.reshape(N_test,nxyz_low)
HH  = HH.reshape(N_test,nxyz_high)


LL  = scaler_Plow.inverse_transform(LL)
SSK = scaler_Slow.inverse_transform(SSK)
VVX = scaler_Vxlow.inverse_transform(VVX)
VVY = scaler_Vylow.inverse_transform(VVY)
VVZ = scaler_Vzlow.inverse_transform(VVZ)
HH  = scaler_Phigh.inverse_transform(HH)


rescaled_LR = LL.reshape(N_test,nx_low,ny_low,nz_low)
rescaled_SK = SSK.reshape(N_test,nx_low,ny_low,nz_low)
rescaled_VX = VVX.reshape(N_test,nx_low,ny_low,nz_low)
rescaled_VY = VVY.reshape(N_test,nx_low,ny_low,nz_low)
rescaled_VZ = VVZ.reshape(N_test,nx_low,ny_low,nz_low)
rescaled_HR = HH.reshape(N_test,nx_high,ny_high,nz_high)

SR_list = []
with torch.no_grad():
    for i in range(N_test):
        #print(rescaled_LR[i].shape)
        # Convert rescaled_LR back to a PyTorch tensor and ensure it's on the correct device
        lr_input = torch.tensor(rescaled_LR[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Unsqueeze to add a batch dimension
        #print(lr_input.shape)
        sk_input = torch.tensor(rescaled_SK[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        vx_input = torch.tensor(rescaled_VX[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        vy_input = torch.tensor(rescaled_VY[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        vz_input = torch.tensor(rescaled_VZ[i], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Generate SR using the model
        # Adjust this part according to your model's specific input requirements
        if model.__class__.__name__ == 'tFUSFormer_5ch':
            # Assuming model expects multiple inputs, adjust as necessary
            sr_output = model(lr_input, sk_input, vx_input, vy_input, vz_input)
        else:
            # If the model only requires LR input
            sr_output = model(lr_input)
        
        # Assuming the output needs to be squeezed to remove the batch dimension if it's singular
        sr_rescaled = sr_output.squeeze(0)
        
        # Append the generated SR to the SR_list
        SR_list.append(sr_rescaled)
        
# Concatenate the list of SR tensors into a single tensor
rescaled_SR = torch.cat(SR_list, dim=0).to('cpu')


###################################################
IoU_vec = np.zeros((N_test))
dist_vec = np.zeros((N_test))


def iou_individual(targets, inputs): 
    smooth = 1.0e-6
    
    # Ensure inputs are tensors
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs).to(torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).to(torch.float32)
    
    tmp_inputs  = torch.zeros_like(inputs)
    tmp_targets = torch.zeros_like(targets)
    #========================
    # FWHM
    #========================
    tmp_inputs[inputs>=0.5*torch.max(inputs)]   = 1
    tmp_targets[targets>=0.5*torch.max(inputs)] = 1

    intersection = (tmp_inputs.flatten() * tmp_targets.flatten()).sum()
    total = (tmp_inputs.flatten() + tmp_targets.flatten()).sum()
    union = total - intersection
    IoU = (intersection + smooth)/(union + smooth)
    return IoU.item()

def dist_individual(targets, inputs):
    # Ensure inputs are tensors
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs).to(torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).to(torch.float32)
    
    # Calculate the indices of the maximum values
    ind1 = torch.argmax(inputs)  # This is a flattened index
    ind2 = torch.argmax(targets)  # This is a flattened index

    # Convert the flattened index to multi-dimensional indices
    ind1_unraveled = torch.tensor(np.unravel_index(ind1.cpu(), inputs.shape)).to(inputs.device)
    ind2_unraveled = torch.tensor(np.unravel_index(ind2.cpu(), targets.shape)).to(targets.device)

    # Calculate the distance
    dist = torch.norm(ind1_unraveled.float() - ind2_unraveled.float(), p=2) / 2
    return dist.item()


folder_path = f'test_results/{model_name}_{test_data_mode}/plot/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for sample in range(N_test):
    LR_slice = rescaled_LR[sample,:,12,:]
    SR_slice = rescaled_SR[sample,:,50,:]
    HR_slice = rescaled_HR[sample,:,50,:]

    plt.figure(figsize=(22,22))
        
    iou_result = iou_individual(rescaled_HR[sample,:,:,:], rescaled_SR[sample,:,:,:])
    IoU_vec[sample] = iou_result
    
    dist_result = dist_individual(rescaled_HR[sample,:,:,:], rescaled_SR[sample,:,:,:])
    dist_vec[sample] = dist_result

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 6.5))
    fig.suptitle('IoU = %f' %iou_result,fontsize=25)
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
    file_path2 = os.path.join(folder_path, 'sample%d.jpg' %sample)
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
folder_path1 = f'test_results/{model_name}_{test_data_mode}/'
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
    if model.__class__.__name__ == 'tFUSFormer_5ch':
        output = model(lr, sk_lr, Vx_lr, Vy_lr, Vz_lr).to(device)
    else:
        output = model(lr).to(device)

# Record end event
end_event.record()

# Wait for events to complete and measure time
torch.cuda.synchronize()
inference_time = start_event.elapsed_time(end_event) / 1000.0  # convert from milliseconds to seconds
print(f"Inference time: {inference_time:.4f} seconds")

file_path5 = os.path.join(folder_path1, 'inference_time.txt')
with open(file_path5, 'w') as f:
    print('=================================================', file=f)
    print('Inference time = ', inference_time, file=f)
    print('=================================================', file=f)  
     
