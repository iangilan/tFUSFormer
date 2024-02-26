import scipy.io
import concurrent.futures
import torch
import unfoldNd
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
import torch.utils.checkpoint as checkpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm, trange
# import EarlyStopping
#from pytorchtools import EarlyStopping

from time import sleep
import collections.abc
from itertools import repeat


def load_data_for_index(i):
    """Load data for a specific index and all prefixes."""
    data_for_i = []
    for prefix in ['1', '2', '3', '4', '5', '6', '9', '10', '11', '12', '13']:
        data_for_i.append(load_data(prefix, i))
    return data_for_i

def load_valid_data_for_index(i):
    """Load data for a specific index and all prefixes."""
    data_for_i = []
    for prefix in ['14']:
        data_for_i.append(load_data(prefix, i))
    return data_for_i

def load_data(prefix, i):
    """Load data for given prefix and index i."""
    
    p_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/p{prefix}_data/pmax_LR_P{prefix}_data{i}.mat'
    p_HR_path = f'/home/mws/BrainUltrasoundSimulation/output/HR/p{prefix}_data/pmax_HR_P{prefix}_data{i}.mat'
    ux_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/p{prefix}_data/ux_LR_P{prefix}_data{i}.mat'
    uy_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/p{prefix}_data/uy_LR_P{prefix}_data{i}.mat'
    uz_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/p{prefix}_data/uz_LR_P{prefix}_data{i}.mat'
    sk_LR_path = f'/home/mws/BrainUltrasoundSimulation/input/sk{prefix}_data/skull_patch{prefix}_data{i}.mat'

    # Load data
    p_LR = scipy.io.loadmat(p_LR_path)['p_max_ROI']
    p_HR = scipy.io.loadmat(p_HR_path)['p_max_ROI']
    ux_LR = scipy.io.loadmat(ux_LR_path)['ux_ROI']
    uy_LR = scipy.io.loadmat(uy_LR_path)['uy_ROI']
    uz_LR = scipy.io.loadmat(uz_LR_path)['uz_ROI']
    sk_LR = scipy.io.loadmat(sk_LR_path)['sk']

    return p_LR, p_HR, ux_LR, uy_LR, uz_LR, sk_LR

# Define the range of file numbers to load
start_index = 1
end_index = 600

# Initialize an empty list to store the matrices
matrices_LR, matrices_HR, ux_matrices_LR, uy_matrices_LR, uz_matrices_LR, sk_matrices_LR = [], [], [], [], [], []
valid_matrices_LR, valid_matrices_HR, valid_ux_matrices_LR, valid_uy_matrices_LR, valid_uz_matrices_LR, valid_sk_matrices_LR = [], [], [], [], [], []

# Use concurrent futures to parallelize file reading
with concurrent.futures.ThreadPoolExecutor() as executor:
    for data_for_i in executor.map(load_data_for_index, range(start_index, end_index+1)):
        for p_LR, p_HR, ux_LR, uy_LR, uz_LR, sk_LR in data_for_i:
            matrices_LR.append(p_LR)
            matrices_HR.append(p_HR)
            ux_matrices_LR.append(ux_LR)
            uy_matrices_LR.append(uy_LR)
            uz_matrices_LR.append(uz_LR)
            sk_matrices_LR.append(sk_LR)        

with concurrent.futures.ThreadPoolExecutor() as executor:
    for data_for_i in executor.map(load_valid_data_for_index, range(start_index, end_index+1)):
        for p_LR, p_HR, ux_LR, uy_LR, uz_LR, sk_LR in data_for_i:
            valid_matrices_LR.append(p_LR)
            valid_matrices_HR.append(p_HR)
            valid_ux_matrices_LR.append(ux_LR)
            valid_uy_matrices_LR.append(uy_LR)
            valid_uz_matrices_LR.append(uz_LR)
            valid_sk_matrices_LR.append(sk_LR)     
            
del p_LR, p_HR, ux_LR, uy_LR, uz_LR, sk_LR, data_for_i          

start_index1 = int((32 - 25) / 2)
end_index1   = start_index1 + 25

start_index2 = int((128 - 100) / 2)
end_index2   = start_index2 + 100

N_samples = np.shape(matrices_LR)[0]
lr_data   = np.zeros((N_samples,25,25,25))
Vx_data   = np.zeros((N_samples,25,25,25))
Vy_data   = np.zeros((N_samples,25,25,25))
Vz_data   = np.zeros((N_samples,25,25,25))
sk_data   = np.zeros((N_samples,25,25,25))
hr_data   = np.zeros((N_samples,100,100,100))

# Extract the center portion of lr_data
for i in range(N_samples):
    lr_data[i,:,:,:] = matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    Vx_data[i,:,:,:] = ux_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    Vy_data[i,:,:,:] = uy_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    Vz_data[i,:,:,:] = uz_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    sk_data[i,:,:,:] = sk_matrices_LR[i][0:25, 0:25, 0:25]
    hr_data[i,:,:,:] = matrices_HR[i][start_index2:end_index2, start_index2:end_index2, start_index2:end_index2]

del matrices_LR,ux_matrices_LR,uy_matrices_LR,uz_matrices_LR,sk_matrices_LR,matrices_HR

valid_N_samples = np.shape(valid_matrices_LR)[0]
valid_lr_data   = np.zeros((valid_N_samples,25,25,25))
valid_Vx_data   = np.zeros((valid_N_samples,25,25,25))
valid_Vy_data   = np.zeros((valid_N_samples,25,25,25))
valid_Vz_data   = np.zeros((valid_N_samples,25,25,25))
valid_sk_data   = np.zeros((valid_N_samples,25,25,25))
valid_hr_data   = np.zeros((valid_N_samples,100,100,100))

for i in range(valid_N_samples):
    valid_lr_data[i,:,:,:] = valid_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    valid_Vx_data[i,:,:,:] = valid_ux_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    valid_Vy_data[i,:,:,:] = valid_uy_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    valid_Vz_data[i,:,:,:] = valid_uz_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    valid_sk_data[i,:,:,:] = valid_sk_matrices_LR[i][0:25, 0:25, 0:25]
    valid_hr_data[i,:,:,:] = valid_matrices_HR[i][start_index2:end_index2, start_index2:end_index2, start_index2:end_index2]

del valid_matrices_LR,valid_ux_matrices_LR,valid_uy_matrices_LR,valid_uz_matrices_LR,valid_sk_matrices_LR,valid_matrices_HR

N_samples_hr, nx_high, ny_high, nz_high = hr_data.shape
N_samples_lr, nx_low, ny_low, nz_low    = lr_data.shape
valid_N_samples_hr, valid_nx_high, valid_ny_high, valid_nz_high = valid_hr_data.shape
valid_N_samples_lr, valid_nx_low, valid_ny_low, valid_nz_low    = valid_lr_data.shape
nxyz_high = nz_high*nx_high*ny_high
nxyz_low  = nz_low*nx_low*ny_low
N = N_samples_lr
valid_N = valid_N_samples_lr

P_high    = hr_data.reshape(N, nxyz_high)
del hr_data
P_low     = lr_data.reshape(N, nxyz_low)
del lr_data
skull_low = sk_data.reshape(N, nxyz_low)
del sk_data
Vx_low    = Vx_data.reshape(N, nxyz_low)
del Vx_data
Vy_low    = Vy_data.reshape(N, nxyz_low)
del Vy_data
Vz_low    = Vz_data.reshape(N, nxyz_low)
del Vz_data

valid_P_high    = valid_hr_data.reshape(valid_N, nxyz_high)
del valid_hr_data
valid_P_low     = valid_lr_data.reshape(valid_N, nxyz_low)
del valid_lr_data
valid_skull_low = valid_sk_data.reshape(valid_N, nxyz_low)
del valid_sk_data
valid_Vx_low    = valid_Vx_data.reshape(valid_N, nxyz_low)
del valid_Vx_data
valid_Vy_low    = valid_Vy_data.reshape(valid_N, nxyz_low)
del valid_Vy_data
valid_Vz_low    = valid_Vz_data.reshape(valid_N, nxyz_low)
del valid_Vz_data

Plow_train = P_low
del P_low
Phigh_train = P_high
del P_high
skull_train = skull_low
del skull_low
Vx_train = Vx_low
Vy_train = Vy_low
Vz_train = Vz_low
del Vx_low, Vy_low, Vz_low
Plow_valid, Plow_test, Phigh_valid, Phigh_test = train_test_split(valid_P_low, valid_P_high, test_size = 0.5, random_state = 1)
skull_valid, skull_test = train_test_split(valid_skull_low, test_size = 0.5, random_state = 1)
Vx_valid, Vx_test = train_test_split(valid_Vx_low, test_size = 0.5, random_state = 1)
Vy_valid, Vy_test = train_test_split(valid_Vy_low, test_size = 0.5, random_state = 1)
Vz_valid, Vz_test = train_test_split(valid_Vz_low, test_size = 0.5, random_state = 1)

N_train = (Plow_train.shape)[0]
N_valid = (Plow_valid.shape)[0]
N_test  = (Plow_test.shape) [0]

#################################################################################################
del valid_P_low, valid_P_high, valid_skull_low, valid_Vx_low, valid_Vy_low, valid_Vz_low
#################################################################################################


## standardize the values of training and testing data from 0 to 1
scaler_Phigh = MinMaxScaler()
scaler_Plow  = MinMaxScaler()
scaler_Slow  = MinMaxScaler()
scaler_Vxlow = MinMaxScaler()
scaler_Vylow = MinMaxScaler()
scaler_Vzlow = MinMaxScaler()


train_P_high_scale    = np.zeros((N_train, 1, nx_high, ny_high, nz_high), dtype='float32')

train_P_low_scale     = np.zeros((N_train, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
train_skull_low_scale = np.zeros((N_train, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
train_Vx_low_scale    = np.zeros((N_train, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
train_Vy_low_scale    = np.zeros((N_train, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
train_Vz_low_scale    = np.zeros((N_train, 1, nx_low,  ny_low,  nz_low ), dtype='float32')

valid_P_high_scale    = np.zeros((N_valid, 1, nx_high, ny_high, nz_high), dtype='float32')

valid_P_low_scale     = np.zeros((N_valid, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
valid_skull_low_scale = np.zeros((N_valid, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
valid_Vx_low_scale    = np.zeros((N_valid, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
valid_Vy_low_scale    = np.zeros((N_valid, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
valid_Vz_low_scale    = np.zeros((N_valid, 1, nx_low,  ny_low,  nz_low ), dtype='float32')

test_P_high_scale    = np.zeros(shape = (N_test, 1, nx_high, ny_high, nz_high), dtype='float32')

test_P_low_scale     = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_skull_low_scale = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_Vx_low_scale    = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_Vy_low_scale    = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_Vz_low_scale    = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')

Phigh_train = scaler_Phigh.fit_transform(Phigh_train)
Phigh_valid = scaler_Phigh.transform(Phigh_valid)
Phigh_test  = scaler_Phigh.transform(Phigh_test)

Plow_train  = scaler_Plow.fit_transform(Plow_train)
Plow_valid  = scaler_Plow.transform(Plow_valid)
Plow_test   = scaler_Plow.transform(Plow_test)

skull_train = scaler_Slow.fit_transform(skull_train)
skull_valid = scaler_Slow.transform(skull_valid)
skull_test  = scaler_Slow.transform(skull_test)

Vx_train = scaler_Vxlow.fit_transform(Vx_train)
Vx_valid = scaler_Vxlow.transform(Vx_valid)
Vx_test  = scaler_Vxlow.transform(Vx_test)

Vy_train = scaler_Vylow.fit_transform(Vy_train)
Vy_valid = scaler_Vylow.transform(Vy_valid)
Vy_test  = scaler_Vylow.transform(Vy_test)

Vz_train = scaler_Vzlow.fit_transform(Vz_train)
Vz_valid = scaler_Vzlow.transform(Vz_valid)
Vz_test  = scaler_Vzlow.transform(Vz_test)


for i in range(N_train):
    train_P_high_scale[i, 0, :, :, :]     = np.reshape(Phigh_train[i, :],(nx_high, ny_high, nz_high))
    train_P_low_scale [i, 0, :, :, :]     = np.reshape(Plow_train[i, :],  (nx_low,  ny_low,  nz_low))
    train_skull_low_scale[i, 0, :, :, :]  = np.reshape(skull_train[i, :], (nx_low,  ny_low,  nz_low))
    train_Vx_low_scale [i, 0, :, :, :]    = np.reshape(Vx_train[i, :],    (nx_low,  ny_low,  nz_low))
    train_Vy_low_scale [i, 0, :, :, :]    = np.reshape(Vy_train[i, :],    (nx_low,  ny_low,  nz_low))
    train_Vz_low_scale [i, 0, :, :, :]    = np.reshape(Vz_train[i, :],    (nx_low,  ny_low,  nz_low))

for i in range(N_valid):
    valid_P_high_scale[i, 0, :, :, :]     = np.reshape(Phigh_valid[i, :], (nx_high, ny_high, nz_high))
    valid_P_low_scale [i, 0, :, :, :]     = np.reshape(Plow_valid[i, :],   (nx_low,  ny_low,  nz_low))
    valid_skull_low_scale[i, 0, :, :, :]  = np.reshape(skull_valid[i, :],  (nx_low,  ny_low,  nz_low))
    valid_Vx_low_scale [i, 0, :, :, :]    = np.reshape(Vx_valid[i, :],     (nx_low,  ny_low,  nz_low))
    valid_Vy_low_scale [i, 0, :, :, :]    = np.reshape(Vy_valid[i, :],     (nx_low,  ny_low,  nz_low))
    valid_Vz_low_scale [i, 0, :, :, :]    = np.reshape(Vz_valid[i, :],     (nx_low,  ny_low,  nz_low))

for i in range(N_test):
    test_P_high_scale[i, 0, :, :, :]     = np.reshape(Phigh_test[i, :], (nx_high, ny_high, nz_high))
    test_P_low_scale [i, 0, :, :, :]     = np.reshape(Plow_test[i, :],   (nx_low,  ny_low,  nz_low))
    test_skull_low_scale[i, 0, :, :, :]  = np.reshape(skull_test[i, :],  (nx_low,  ny_low,  nz_low))
    test_Vx_low_scale [i, 0, :, :, :]    = np.reshape(Vx_test[i, :],     (nx_low,  ny_low,  nz_low))
    test_Vy_low_scale [i, 0, :, :, :]    = np.reshape(Vy_test[i, :],     (nx_low,  ny_low,  nz_low))
    test_Vz_low_scale [i, 0, :, :, :]    = np.reshape(Vz_test[i, :],     (nx_low,  ny_low,  nz_low))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 4
load_start = time.time()
#=========================================================================================================
class SRCNNDataset(Dataset):
    def __init__(self, sim_data1, sim_data2, sim_data3, sim_data4, sim_data5, labels):
        self.sim_data1 = sim_data1
        self.sim_data2 = sim_data2
        self.sim_data3 = sim_data3
        self.sim_data4 = sim_data4
        self.sim_data5 = sim_data5
        self.labels = labels

    def __len__(self):
        return (len(self.sim_data1))

    def __getitem__(self, index):
        sim1  = self.sim_data1[index]
        sim2  = self.sim_data2[index]
        sim3  = self.sim_data3[index]
        sim4  = self.sim_data4[index]
        sim5  = self.sim_data5[index]
        label = self.labels[index]
        return (torch.tensor(sim1, dtype=torch.float32), torch.tensor(sim2,  dtype=torch.float32),
                torch.tensor(sim3, dtype=torch.float32), torch.tensor(sim4,  dtype=torch.float32),
                torch.tensor(sim5, dtype=torch.float32), torch.tensor(label, dtype=torch.float32))

train_ds = SRCNNDataset(train_P_low_scale, train_skull_low_scale, train_Vx_low_scale, train_Vy_low_scale, train_Vz_low_scale, train_P_high_scale)
del train_P_low_scale, train_skull_low_scale, train_Vx_low_scale, train_Vy_low_scale, train_Vz_low_scale, train_P_high_scale
valid_ds = SRCNNDataset(valid_P_low_scale, valid_skull_low_scale, valid_Vx_low_scale, valid_Vy_low_scale, valid_Vz_low_scale, valid_P_high_scale)
del valid_P_low_scale, valid_skull_low_scale, valid_Vx_low_scale, valid_Vy_low_scale, valid_Vz_low_scale,  valid_P_high_scale
test_ds  = SRCNNDataset(test_P_low_scale, test_skull_low_scale, test_Vx_low_scale, test_Vy_low_scale, test_Vz_low_scale, test_P_high_scale)
del test_P_low_scale, test_skull_low_scale, test_Vx_low_scale, test_Vy_low_scale, test_Vz_low_scale, test_P_high_scale


train_dl = DataLoader(train_ds, batch_size=batch_size)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)
test_dl  = DataLoader(test_ds,  batch_size=batch_size)
load_end = time.time()
print('Data loading time = ',load_end-load_start,'s')
#  73.60881495475769 s
print('N_train = ', N_train)
print('N_valid = ', N_valid)
print('N_test  = ', N_test)
#####################################################################################
# helper functions
#####################################################################################
# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


'''
reference: http://www.multisilicon.com/blog/a25332339.html
'''
class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of PixelShuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_height, in_width, in_depth = input.size()            
        nOut = channels // self.scale ** 3
        out_height = in_height * self.scale
        out_width  = in_width  * self.scale
        out_depth  = in_depth  * self.scale
        
        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_height, in_width, in_depth)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_height, out_width, out_depth)
    

def iou(targets, inputs): 
    smooth = 1.0e-6
    targets = targets.cpu().detach().numpy()
    inputs  = inputs.cpu().detach().numpy()
    #========================
    # FWHM
    #========================
    inputs[inputs>=0.5]   = 1
    targets[targets>=0.5] = 1
    inputs[inputs<0.5]    = 0
    targets[targets<0.5]  = 0
    #========================
    #==========================================
    # FWHM
    #==========================================
    # inputs [inputs  >= 0.5*np.max(inputs) ] = 1
    # targets[targets >= 0.5*np.max(targets)] = 1
    # inputs [inputs  <  0.5*np.max(inputs) ] = 0
    # targets[targets <  0.5*np.max(targets)] = 0
    #==========================================      
    targets = targets.squeeze(1)
    inputs  = inputs.squeeze(1)
    
    targets = targets.flatten()
    inputs = inputs.flatten()
    
    intersection = (inputs * targets).sum()
    total = (inputs + targets).sum()
    union = total - intersection 
    IoU = (intersection + smooth)/(union + smooth)

    return IoU

loss_func = nn.MSELoss()    

#####################################################################################
# helper functions
#####################################################################################
# From PyTorch internals
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 3d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor: Tensor, mean: float = 0., std: float = 1., a: float = -2., b: float = 2.) -> Tensor:
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, D, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, D, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W, D):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        D (int): Depth of image
    Returns:
        x: (B, H, W, D, C)
    """
    B = int(windows.shape[0] / (H * W *D/ window_size / window_size/ window_size))
    x = windows.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, D, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., rpb=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww, Wd
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.rpb = rpb

        # define a parameter table of relative position bias
        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2*window_size[0] - 1)*(2 * window_size[1] - 1)*(2 * window_size[2] - 1), num_heads))
            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            coords_d = torch.arange(self.window_size[2])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w,coords_d]))
            coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wd
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wd, Wh*Ww*Wd
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wd, Wh*Ww*Wd, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wd, Wh*Ww*Wd

            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C) N=H*W*D
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww*Wd, Wh*Ww*Wd) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpb:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1]*self.window_size[2], self.window_size[0] * self.window_size[1]*self.window_size[2],-1)  # Wh*Ww*Wd, Wh*Wd*Ww, nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wd,Wd*Ww*Wh
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=5, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, rpb = True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.rpb = rpb
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_3tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, rpb=self.rpb)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W, D= x_size
        img_mask = torch.zeros((1, H, W, D, 1))  # 1 H W D 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        d_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for d in d_slices:
                    img_mask[:, h, w, d, :] = cnt
                    cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size*self.window_size)
        attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2))
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, x_size):
        H, W, D = x_size
        B, L, C = x.shape


        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, D, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size*self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, D)  # B H' W' D' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W * D, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=True, rpb=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 rpb = rpb)
            for i in range(depth)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=100, patch_size=4, resi_connection='1conv', rpb=True):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint,
                                         rpb=rpb)

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(nn.Conv3d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv3d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv3d(dim // 4, dim, 3, 1, 1))
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed3D(nn.Module):
    r""" 3D Volume to Patch Embedding
    Args:
        vol_size (int): Volume size.  Default: 64.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=100, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, level="second"):
        super().__init__()
        img_size = to_3tuple(img_size)
        if patch_size>1:
            self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = None
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1],img_size[2] // patch_size[2] ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]* patches_resolution[2]
        self.level = level
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        if self.proj:
            proj_x = self.proj(x)
            x = proj_x.flatten(2).transpose(1,2)
        else:
            if x.shape[1]==self.embed_dim:
                x = x.flatten(2).transpose(1, 2)
            else:
                x = x#.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed3D(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=100, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2]//patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]* patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.patch_size = patch_size

    def forward(self, x, x_size):
        B, HWD, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1], x_size[2])  # B Ph*Pw*Pd C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # check if scale is power of 2
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, (scale ** 3) * num_feat, 3, 1, 1))
                m.append(PixelShuffle3d(scale))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, (scale ** 3) * num_feat, 3, 1, 1))
            m.append(PixelShuffle3d(scale))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        #m.append(nn.Conv2d(num_feat, (scale ** 3) * num_out_ch, 3, 1, 1))
        m.append(nn.Conv3d(num_feat, (scale ** 3) * num_out_ch, 3, 1, 1))
        #m.append(nn.PixelShuffle(scale))
        m.append(PixelShuffle3d(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SuperFormer(nn.Module):
    r""" SwinIR
        A PyTorch impl of : `SwinIR: Image Restoration Using Swin Transformer`, based on Swin Transformer.
        Implementación promediando las representaciones a la entrada del transformer.
    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=25, patch_size=1, in_chans=1,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=5, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, rpb=True ,patch_norm=True,
                 use_checkpoint=True, upscale=4, img_range=1., upsampler='', resi_connection='1conv',
                 output_type = "residual",num_feat=64,**kwargs):
        super(SuperFormer, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = num_feat
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)
            #self.mean = torch.zeros(1, num_in_ch, 1, 1, 1) # not sure
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.patch_size = patch_size

        #####################################################################################################
        ################################### 3D shallow feature extraction ###################################
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)
        #####################################################################################################
        ################################### 3D Deep Feature Extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.rpb = rpb
        self.output_type = output_type
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio


        self.patch_embed_features = PatchEmbed3D(
            img_size = img_size, patch_size = patch_size, in_chans = embed_dim, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None, level = "first")

        self.patch_embed_volume = PatchEmbed3D(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None, level = "first")

        num_patches = self.patch_embed_volume.num_patches
        patches_resolution = self.patch_embed_volume.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1],
                                           patches_resolution[2]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=1,
                         resi_connection=resi_connection,
                         rpb = self.rpb
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':

            self.conv_after_body = nn.Sequential(nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3D high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            #self.upsample = nn.Upsample(scale_factor=2, mode='nearest')#nn.Upsample(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1  = nn.Conv3d(num_feat, num_feat,   3, 1, 1)
            self.conv_up2  = nn.Conv3d(num_feat, num_feat,   3, 1, 1)
            self.conv_hr   = nn.Conv3d(num_feat, num_feat,   3, 1, 1)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu     = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            if self.patch_size>1:
                self.upsample_feat = nn.Sequential(nn.ConvTranspose3d(embed_dim,embed_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                                  nn.Conv3d(embed_dim, embed_dim, 3, 1, 1),
                                                  nn.InstanceNorm3d(embed_dim),
                                                  nn.LeakyReLU(inplace=True))
            if self.output_type=='residual':
                self.conv_last = nn.Conv3d(embed_dim, num_out_ch, 3, 1, 1)
            else:
                self.conv_before_last = nn.Sequential(nn.Conv3d(embed_dim, num_feat,3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
                self.conv_last= nn.Conv3d(num_feat, num_out_ch, 1, 1, 0)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w, d = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        mod_pad_d = (self.window_size - d % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'replicate')

        return x

#     def forward_features(self, x, x_feat):
#         print('x.shape in forward_features (weird)',x.shape) # 4 1 50 50 50
#         print('x_feat.shape in forward_features (weird)',x_feat.shape) # 4 96 50 50 50
#         if self.patch_size>1:
#             x_size = self.patches_resolution
#             print('x_size if self.patch_size>1', x_size)
#         else:
#             x_size = (x.shape[2], x.shape[3], x.shape[4])
#             print('x_size if not self.patch_size>1', x_size)
#         print('x_size in forward_features', x_size)
#         x_feat = self.patch_embed_features(x_feat)
#         x_vol= self.patch_embed_volume(x)
#         print('x_feat in forward_features!!!!!!!!!!!', x_feat)
#         print('x_vol in forward_features!!!!!!!!!!', x_vol)

#         if self.ape:
#             x_feat = x_feat + self.absolute_pos_embed
#             x_vol = x_vol + self.absolute_pos_embed
#         x_feat = self.pos_drop(x_feat)
#         x_vol = self.pos_drop(x_vol)

#         for layer in self.layers:
#             x_feat = layer(x_feat, x_size)
#             x_vol = layer(x_vol, x_size)

#         x_feat = self.norm(x_feat)
#         x_vol = self.norm(x_vol)
#         x_feat = self.patch_unembed(x_feat, x_size)
#         x_vol = self.patch_unembed(x_vol, x_size)
#         return x_feat, x_vol


    def forward_features(self, x):
        if self.patch_size>1:
            x_size = self.patches_resolution
        else:
            x_size = (x.shape[2], x.shape[3], x.shape[4])
        x_feat  = self.patch_embed_features(x)

        #x_vol= self.patch_embed_volume(x)

        if self.ape:
            x_feat  = x_feat + self.absolute_pos_embed
            #x_vol = x_vol + self.absolute_pos_embed
        x_feat = self.pos_drop(x_feat)
        #x_vol = self.pos_drop(x_vol)

        for layer in self.layers:
            x_feat  = layer(x_feat, x_size)
            #x_vol = layer(x_vol, x_size)

        x_feat = self.norm(x_feat)

        #x_vol  = self.norm(x_vol)
        x_feat = self.patch_unembed(x_feat, x_size)

        #x_vol  = self.patch_unembed(x_vol, x_size)
        return x_feat #, x_vol



    def forward(self, x, x2, x3, x4, x5):
        H, W, D = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        self.mean2 = self.mean.type_as(x2)
        self.mean3 = self.mean.type_as(x3)
        self.mean4 = self.mean.type_as(x4)
        self.mean5 = self.mean.type_as(x5)
        x = (x - self.mean) * self.img_range
        x2 = (x2 - self.mean2) * self.img_range
        x3 = (x3 - self.mean3) * self.img_range
        x4 = (x4 - self.mean4) * self.img_range
        x5 = (x5 - self.mean5) * self.img_range

        ####################################################
        # newly added since it wasn't in the original code
        ####################################################
        # x_first = self.conv_first(x)
        # print('x_first.shape ',x_first.shape)
        # x_feat, x_vol = self.forward_features(x, x_first)
        # print('x_feat.shape ',x_feat.shape)
        # print('x_vol.shape ',x_vol.shape)
        ####################################################
        if self.upsampler == 'pixelshuffle': # for now only pixelshuffle works!!!
            x = self.conv_first(x)
            x2 = self.conv_first(x2)
            x3 = self.conv_first(x3)
            x4 = self.conv_first(x4)
            x5 = self.conv_first(x5)
            #x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_after_body(self.forward_features(x)) + self.conv_after_body(self.forward_features(x2))
            + self.conv_after_body(self.forward_features(x3)) + self.conv_after_body(self.forward_features(x4))
            self.conv_after_body(self.forward_features(x5)) + + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            if self.patch_size>1:
                x_feat, x_vol = self.forward_features(x) # not sure which one is right!!!!!!!!!!!!!!!!!!
                #x_feat, x_vol = self.forward_features(x_first)
                res_deep_feat = self.conv_after_body(x_feat)
                res_deep_vol = self.conv_after_body(x_vol)
                res_deep = (res_deep_feat + res_deep_vol)/2
                res = self.upsample_feat(res_deep)
                res = res + x_first
            else:
                res = self.conv_after_body(self.forward_features(x_first)) + x_first
                #res = self.conv_after_body(self.forward_features(x,x_first)) + x_first
            if self.output_type == 'residual':
                x = x + self.conv_last(res)
            else:
                x = self.conv_last(self.conv_before_last(res))

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale, :D*self.upscale]


    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops

#model = SuperFormer(upsampler='pixelshuffle').to(device)
#model = SuperFormer(upsampler='pixelshuffledirect').to(device)
model = SuperFormer(upsampler='nearest+conv').to(device)

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

    final_loss = running_loss / len(data_dl.dataset)
    final_iou = running_iou / int(len(train_ds)/data_dl.batch_size)

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

    final_loss = running_loss / len(data_dl.dataset)
    final_iou = running_iou / int(len(data_dl)/data_dl.batch_size)

    return final_loss, final_iou

optimizer = optim.Adam(model.parameters(), lr=0.0002)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = optim.Adam(model.parameters(), lr=0.000005)
num_epochs = 100 # 51 done

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
    #print(f'Train PSNR: {train_epoch_psnr:.3f}, Val PSNR: {val_epoch_psnr:.3f}, Time: {end-start1:.2f} sec, Total Time: {end-start:.2f} sec')

torch.save(model, '/home/mws/hatsr/model/more_4x_5ch_unforeseen_model.pth')
torch.save(model, '/home/mws/hatsr/model/more_4x_5ch_unforeseen_model.pt')
model_scripted = torch.jit.script(model) # TorchScript 형식으로 내보내기
model_scripted.save('/home/mws/hatsr/model/superformer_4x_5ch_unforeseen_model.pt') # 저장하기
model = torch.jit.load('/home/mws/hatsr/model/superformer_4x_5ch_unforeseen_model.pt')
model.eval()

#model = torch.load('/home/mws/hatsr/model/superformer_4x_5ch_unforeseen_model.pt')
model = torch.load('/home/mws/hatsr/model/more_4x_5ch_unforeseen_model.pt')

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss[:67], color='orange', label='train loss')
plt.plot(val_loss[:67], color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
#plt.savefig('plot/train_valid_loss.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

# IoU plots
plt.figure(figsize=(10, 7))
plt.plot(train_iou[:67], color='green', label='train IoU')
#plt.plot(val_iou, color='blue', label='validataion IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU (%)')
plt.legend()
#plt.savefig('plot/train_IoU.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

LR_list, HR_list, SR_list = [], [], []
model.eval()
# with torch.no_grad():
#     sample_low_res_image = sample_low_res_image.to(device)  # Assuming sample_low_res_image is a single low-resolution image
#     super_resolved_image = generator(sample_low_res_image)

with torch.no_grad():
    for lr, sk_lr, Vx_lr, Vy_lr, Vz_lr, hr in test_dl:
        LR_list.append(lr.squeeze(1))
        HR_list.append(hr.squeeze(1))
        SR_list.append(model(lr.to(device),sk_lr.to(device), Vx_lr.to(device), Vy_lr.to(device), Vz_lr.to(device)).squeeze(1))

# Convert lists of tensors into single tensors
LR = torch.cat(LR_list, dim=0)  # This will be of shape [18, 4, 25, 25, 25]
HR = torch.cat(HR_list, dim=0)
SR = torch.cat(SR_list, dim=0)

###################################################
# rescaling SR
###################################################
SS=torch.zeros((N_test,nx_high,ny_high,nz_high))
LL=torch.zeros((N_test,nx_low,ny_low,nz_low))
HH=torch.zeros((N_test,nx_high,ny_high,nz_high))
# SS=torch.zeros((N_test,nx_high,ny_high,nz_high))
# LL=torch.zeros((N_test,nx_low,ny_low,nz_low))
# HH=torch.zeros((N_test,nx_high,ny_high,nz_high))

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

####################################
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

    import os
    folder_path = 'plot/4x_5ch_unforeseen/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    iou_individual = iou_individual(rescaled_HR[sample,:,:,:], rescaled_SR[sample,:,:,:])
    IoU_vec[sample] = iou_individual

    dist_individual = dist_individual(rescaled_HR[sample,:,:,:], rescaled_SR[sample,:,:,:])
    dist_vec[sample] = dist_individual

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(20, 6.5))
    fig.suptitle('IoU = %f' %iou_individual,fontsize=25)
    colorscheme = 'seismic' # jet, seismic, turbo

    axs[0].imshow(LR_slice, cmap=colorscheme)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].imshow(SR_slice, cmap=colorscheme)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(HR_slice, cmap=colorscheme)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[0].set_title(f'LR max = {np.max(rescaled_LR[sample,:,:,:]):.4f}', fontsize=20)
    axs[1].set_title(f'SR max = {np.max(rescaled_SR[sample,:,:,:]):.4f}', fontsize=20)
    axs[2].set_title(f'HR max = {np.max(rescaled_HR[sample,:,:,:]):.4f}', fontsize=20)
    plt.subplots_adjust(wspace = 0.01, hspace = 0.01)

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
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_path1 = os.path.join(folder_path, 'iou_results.txt')

with open(file_path1, 'w') as f:
    print('=================================================', file=f)
    print('IoU_mean for N_test cases   = ', iou_mean, file=f)
    print('IoU_median for N_test cases = ', iou_median, file=f)
    print('=================================================', file=f)

file_path3 = os.path.join(folder_path, 'dist_results.txt')

with open(file_path3, 'w') as f:
    print('=================================================', file=f)
    print('dist_mean for N_test cases   = ', dist_mean, file=f)
    print('dist_median for N_test cases = ', dist_median, file=f)
    print('=================================================', file=f)


####################################
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

    import os
    folder_path = 'plot_new/4x_5ch_unforeseen/'
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
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_path1 = os.path.join(folder_path, 'iou_results.txt')

with open(file_path1, 'w') as f:
    print('=================================================', file=f)
    print('IoU_mean for N_test cases   = ', iou_mean, file=f)
    print('IoU_median for N_test cases = ', iou_median, file=f)
    print('=================================================', file=f)
   
file_path3 = os.path.join(folder_path, 'dist_results.txt')

with open(file_path3, 'w') as f:
    print('=================================================', file=f)
    print('dist_mean for N_test cases   = ', dist_mean, file=f)
    print('dist_median for N_test cases = ', dist_median, file=f)
    print('=================================================', file=f)    

file_path4 = os.path.join(folder_path, 'IoU_vec.txt')
with open(file_path4, 'w') as file:
    for value in IoU_vec:
        file.write(str(value) + '\n')
        
np.save(folder_path + 'LR.npy', rescaled_LR)
np.save(folder_path + 'SR.npy', rescaled_SR)
np.save(folder_path + 'HR.npy', rescaled_HR)        

np.save(folder_path + 'LR.npy', rescaled_LR)
np.save(folder_path + 'SR.npy', rescaled_SR)
np.save(folder_path + 'HR.npy', rescaled_HR)

# Get one batch from test_dl
lr, sk_lr, Vx_lr, Vy_lr, Vz_lr, hr = next(iter(test_dl))

# Inference time check
model = torch.load('/home/mws/hatsr/model/more_4x_5ch_unforeseen_model.pt')

model = model.to(device)
lr    = lr.to(device)
sk_lr = sk_lr.to(device)
Vx_lr = Vx_lr.to(device)
Vy_lr = Vy_lr.to(device)
Vz_lr = Vz_lr.to(device)

# Start time
start_time = time.time()
with torch.no_grad():
    output = model(lr, sk_lr, Vx_lr, Vy_lr, Vz_lr)
               # End time
end_time = time.time()

# Calculate the inference time
inference_time = end_time - start_time
print(f"Inference time: {inference_time:.4f} seconds")

# make sure everything is on the GPU
model = model.to(device)
lr    = lr.to(device)
sk_lr = sk_lr.to(device)
Vx_lr = Vx_lr.to(device)
Vy_lr = Vy_lr.to(device)
Vz_lr = Vz_lr.to(device)

# Create CUDA events
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Record start event
start_event.record()

# Perform inference
with torch.no_grad():
    output = model(lr, sk_lr, Vx_lr, Vy_lr, Vz_lr)

# Record end event
end_event.record()

# Wait for events to complete and measure time
torch.cuda.synchronize()
inference_time = start_event.elapsed_time(end_event) / 1000  # convert from milliseconds to seconds
print(f"Inference time: {inference_time:.4f} seconds")
