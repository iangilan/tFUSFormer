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
from sklearn.decomposition import PCA

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



def register_conv_first_hook(model, hook_fn):
    # Access the conv_first layer directly by its attribute name
    hook = model.conv_first.register_forward_hook(hook_fn)
    return hook  # Return the hook handle for potential removal later

# Define your hook function
def hook_fn(module, input, output):
    # This function will be executed every time a forward pass occurs through the hooked layer
    print(f"Output shape of conv_first layer: {output.shape}")

# Example usage
#model = tFUSFormer_5ch()  # Assuming model is already created

# Register the hook to the conv_first layer
hook = register_conv_first_hook(model, hook_fn)


LR_list, HR_list, SR_list = [], [], []
with torch.no_grad():
    for lr, sk_lr, Vx_lr, Vy_lr, Vz_lr, hr in unforeseen_test_dl if test_data_mode == 'unseen' else test_dl:
        LR_list.append(lr.squeeze(1))
        HR_list.append(hr.squeeze(1))
        if model.__class__.__name__ == 'tFUSFormer_5ch':
            SR_list.append(model(lr.to(device),sk_lr.to(device), Vx_lr.to(device), Vy_lr.to(device), Vz_lr.to(device)).squeeze(1))
        else:    
            SR_list.append(model(lr.to(device)).squeeze(1))


hook.remove()


'''
def get_intermediate_layer(module, input, output):
    global inter_out
    inter_out.append(output)

# Assuming 'conv2d_9' is the name of the layer you're interested in
#layer_name = 'RSTB'
#hook = getattr(model, layer_name).register_forward_hook(get_intermediate_layer)

# Example: Accessing the first layer of a specific type in a ModuleList
for layer in model.layers:
    if isinstance(layer, RSTB):
        hook = layer.register_forward_hook(get_intermediate_layer)
        break  # Assuming you only want to hook to the first RSTB layer found
'''
# Initialize list to store outputs
print('SR_list.shape:', len(SR_list))
print('SR_list[0].shape', SR_list[0].shape)


for i, tensor in enumerate(SR_list):
    print(f"Tensor {i} shape: {tensor.shape}")

all_tensors_concatenated = torch.cat(SR_list, dim=0)

# Verify the shape
print("Concatenated Tensor Shape:", all_tensors_concatenated.shape)

inter_out = all_tensors_concatenated.cpu().numpy()


print('inter_out.shape2: ', inter_out.shape)

# Reshape for PCA as before
n1, h1, w1, c1 = inter_out.shape
inter_out = inter_out.reshape(-1, h1*w1*c1)
print('inter_out.shape3: ', inter_out.shape)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=400, svd_solver='arpack')
inter_out_pca = pca.fit_transform(inter_out)
print(inter_out_pca.shape)

# Save the PCA-transformed features
np.save('../mda/results/tFUSFormer5ch_feature_test.npy', inter_out)


# For the tasks below, five datasets analysed in the manuscript will be automatically loaded. 
# However, you can upload your own dataset, and analyze it using MDA
# Our data were saved as .npy file to reduce the data size (normally .csv file needs more disk space). 
# However, .csv or other type of files can also be loaded and analyzed using MDA


# Load all necessary python packages needed for the reported analyses
# in our manuscript
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

#%matplotlib inline

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import matplotlib.pyplot as plt
import scipy
import scipy.io as sio
import sklearn
import umap
import pandas as pd
from umap.parametric_umap import ParametricUMAP
import numpy as np
from mda import *

# Font size for all the MDA visualizations shown below   
FS = 16

# Number of neighbors in MDA analyses
neighborNum = 5

# Load feature data extracted by the SRGAN at umsampling block from test images
testDataFeatures = np.load('../mda/results/tFUSFormer5ch_feature_test.npy')
# Load data labels (target high resolution images) corresponding to low resolution test images
Y = np.load('../data/SR/y_test.npy')
# Reshape the target images into vectors so that they can be analyzed by MDA 
Y = Y.reshape(Y.shape[0],-1)
# Load output images prediced by the SRGAN
Y_pred = np.load('../data/SR/y_test_pred_trained.npy')
# Reshape the predicted output images into vectors so that they can be analyzed by MDA 
Y_pred = Y_pred.reshape(Y_pred.shape[0],-1)

# Create color map for MDA visualization from the target manifold topology
clusterIdx = discoverManifold(Y, neighborNum)
# Compute the outline of the output manifold
clusterIdx_pred = discoverManifold(Y_pred, neighborNum)
# Use the outline of the output manifold to generate the MDA visualization of the SRGAN features
Yreg = mda(testDataFeatures,clusterIdx_pred)   

# Plot the MDA results
plt.figure(1)
plt.scatter(Yreg[:,0],Yreg[:,1],c=clusterIdx.T, cmap='jet', s=5)
plt.xlabel("MDA1")
plt.ylabel("MDA2")
plt.title('MDA visualization of the SRGAN features for superresolution task')
'''
