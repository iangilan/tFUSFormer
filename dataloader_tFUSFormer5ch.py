import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
import concurrent.futures
import scipy 
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# Use the configurations
batch_size = config.batch_size

def load_data_for_index(i):
    """Load data for a specific index and all prefixes."""
    data_for_i = []
    for prefix in ['1', '2', '3', '4', '5', '6', '9', '10', '11', '12', '13', '14']:
        data_for_i.append(load_data(prefix, i))
    return data_for_i

def load_unforeseen_data_for_index(i):
    """Load data for a specific index and all prefixes."""
    unforeseen_data_for_i = []
    for prefix in ['13', '17', '22']:
        unforeseen_data_for_i.append(load_unforeseen_data(prefix, i))
    return unforeseen_data_for_i


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

def load_unforeseen_data(prefix, i):
    """Load unforeseen data for given prefix and index i."""
    
    p_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/sk{prefix}_data/pmax_LR_P{prefix}_data{i}.mat'
    p_HR_path = f'/home/mws/BrainUltrasoundSimulation/output/HR/sk{prefix}_data/pmax_HR_P{prefix}_data{i}.mat'
    ux_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/sk{prefix}_data/ux_LR_P{prefix}_data{i}.mat'
    uy_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/sk{prefix}_data/uy_LR_P{prefix}_data{i}.mat'
    uz_LR_path = f'/home/mws/BrainUltrasoundSimulation/output/LR/sk{prefix}_data/uz_LR_P{prefix}_data{i}.mat'
    sk_LR_path = f'/home/mws/BrainUltrasoundSimulation/input/sk_data_old/test_sk{prefix}_data/skull_patch{prefix}_data{i}.mat'

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
end_index = 10#600

# Initialize an empty list to store the matrices
matrices_LR, matrices_HR, ux_matrices_LR, uy_matrices_LR, uz_matrices_LR, sk_matrices_LR = [], [], [], [], [], []
unforeseen_matrices_LR, unforeseen_matrices_HR, unforeseen_ux_matrices_LR, unforeseen_uy_matrices_LR, unforeseen_uz_matrices_LR, unforeseen_sk_matrices_LR = [], [], [], [], [], []

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

unforeseen_N_test = np.shape(unforeseen_matrices_LR)[0]
unforeseen_lr_data   = np.zeros((unforeseen_N_test,25,25,25))
unforeseen_Vx_data   = np.zeros((unforeseen_N_test,25,25,25))
unforeseen_Vy_data   = np.zeros((unforeseen_N_test,25,25,25))
unforeseen_Vz_data   = np.zeros((unforeseen_N_test,25,25,25))
unforeseen_sk_data   = np.zeros((unforeseen_N_test,25,25,25))
unforeseen_hr_data   = np.zeros((unforeseen_N_test,100,100,100))

# Extract the center portion of lr_data
for i in range(N_samples):
    lr_data[i,:,:,:] = matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    Vx_data[i,:,:,:] = ux_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    Vy_data[i,:,:,:] = uy_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    Vz_data[i,:,:,:] = uz_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    sk_data[i,:,:,:] = sk_matrices_LR[i][0:25, 0:25, 0:25]
    hr_data[i,:,:,:] = matrices_HR[i][start_index2:end_index2, start_index2:end_index2, start_index2:end_index2]

for i in range(unforeseen_N_test):
    unforeseen_lr_data[i,:,:,:] = unforeseen_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    unforeseen_Vx_data[i,:,:,:] = unforeseen_ux_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    unforeseen_Vy_data[i,:,:,:] = unforeseen_uy_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    unforeseen_Vz_data[i,:,:,:] = unforeseen_uz_matrices_LR[i][start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
    unforeseen_sk_data[i,:,:,:] = unforeseen_sk_matrices_LR[i][0:25, 0:25, 0:25]
    unforeseen_hr_data[i,:,:,:] = unforeseen_matrices_HR[i][start_index2:end_index2, start_index2:end_index2, start_index2:end_index2]

N_samples_hr, nx_high, ny_high, nz_high = hr_data.shape
N_samples_lr, nx_low, ny_low, nz_low    = lr_data.shape
nxyz_high = nz_high*nx_high*ny_high
nxyz_low  = nz_low*nx_low*ny_low
N = N_samples_lr

P_high    = hr_data.reshape(N, nxyz_high)
P_low     = lr_data.reshape(N, nxyz_low)
skull_low = sk_data.reshape(N, nxyz_low)
Vx_low    = Vx_data.reshape(N, nxyz_low)
Vy_low    = Vy_data.reshape(N, nxyz_low)
Vz_low    = Vz_data.reshape(N, nxyz_low)

unforeseen_P_high    = unforeseen_hr_data.reshape(unforeseen_N_test, nxyz_high)
unforeseen_P_low     = unforeseen_lr_data.reshape(unforeseen_N_test, nxyz_low)
unforeseen_skull_low = unforeseen_sk_data.reshape(unforeseen_N_test, nxyz_low)
unforeseen_Vx_low    = unforeseen_Vx_data.reshape(unforeseen_N_test, nxyz_low)
unforeseen_Vy_low    = unforeseen_Vy_data.reshape(unforeseen_N_test, nxyz_low)
unforeseen_Vz_low    = unforeseen_Vz_data.reshape(unforeseen_N_test, nxyz_low)

Plow_train = P_low
Phigh_train = P_high
skull_train = skull_low
Vx_train = Vx_low
Vy_train = Vy_low
Vz_train = Vz_low

Plow_train, Plow_valid, Phigh_train, Phigh_valid = train_test_split(P_low, P_high, test_size = 0.2, random_state = 1)
skull_train, skull_valid = train_test_split(skull_low, test_size = 0.2, random_state = 1)
Vx_train, Vx_valid = train_test_split(Vx_low, test_size = 0.2, random_state = 1)
Vy_train, Vy_valid = train_test_split(Vy_low, test_size = 0.2, random_state = 1)
Vz_train, Vz_valid = train_test_split(Vz_low, test_size = 0.2, random_state = 1)

Plow_valid, Plow_test, Phigh_valid, Phigh_test = train_test_split(Plow_valid, Phigh_valid, test_size = 0.5, random_state = 1)
skull_valid, skull_test = train_test_split(skull_valid, test_size = 0.5, random_state = 1)
Vx_valid, Vx_test = train_test_split(Vx_valid, test_size = 0.5, random_state = 1)
Vy_valid, Vy_test = train_test_split(Vy_valid, test_size = 0.5, random_state = 1)
Vz_valid, Vz_test = train_test_split(Vz_valid, test_size = 0.5, random_state = 1)


N_train = (Plow_train.shape)[0]
N_valid = (Plow_valid.shape)[0]
N_test  = (Plow_test.shape)[0]


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

test_P_high_scale    = np.zeros((N_test, 1, nx_high, ny_high, nz_high), dtype='float32')
test_P_low_scale     = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_skull_low_scale = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_Vx_low_scale    = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_Vy_low_scale    = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
test_Vz_low_scale    = np.zeros((N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')

unforeseen_test_P_high_scale    = np.zeros((unforeseen_N_test, 1, nx_high, ny_high, nz_high), dtype='float32')
unforeseen_test_P_low_scale     = np.zeros((unforeseen_N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
unforeseen_test_skull_low_scale = np.zeros((unforeseen_N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
unforeseen_test_Vx_low_scale    = np.zeros((unforeseen_N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
unforeseen_test_Vy_low_scale    = np.zeros((unforeseen_N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')
unforeseen_test_Vz_low_scale    = np.zeros((unforeseen_N_test, 1, nx_low,  ny_low,  nz_low ), dtype='float32')


Phigh_train = scaler_Phigh.fit_transform(Phigh_train)
Phigh_valid = scaler_Phigh.transform(Phigh_valid)
Phigh_test = scaler_Phigh.transform(Phigh_test)

Plow_train  = scaler_Plow.fit_transform(Plow_train)
Plow_valid  = scaler_Plow.transform(Plow_valid)
Plow_test  = scaler_Plow.transform(Plow_test)

skull_train = scaler_Slow.fit_transform(skull_train)
skull_valid = scaler_Slow.transform(skull_valid)
skull_test = scaler_Slow.transform(skull_test)

Vx_train = scaler_Vxlow.fit_transform(Vx_train)
Vx_valid = scaler_Vxlow.transform(Vx_valid)
Vx_test = scaler_Vxlow.transform(Vx_test)

Vy_train = scaler_Vylow.fit_transform(Vy_train)
Vy_valid = scaler_Vylow.transform(Vy_valid)
Vy_test = scaler_Vylow.transform(Vy_test)

Vz_train = scaler_Vzlow.fit_transform(Vz_train)
Vz_valid = scaler_Vzlow.transform(Vz_valid)
Vz_test = scaler_Vzlow.transform(Vz_test)



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


#=========================================================================================================
class tFUSFormer5chDataset(Dataset):
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

train_ds = tFUSFormer5chDataset(train_P_low_scale, train_skull_low_scale, train_Vx_low_scale, train_Vy_low_scale, train_Vz_low_scale, train_P_high_scale)
valid_ds = tFUSFormer5chDataset(valid_P_low_scale, valid_skull_low_scale, valid_Vx_low_scale, valid_Vy_low_scale, valid_Vz_low_scale, valid_P_high_scale)
test_ds  = tFUSFormer5chDataset(test_P_low_scale, test_skull_low_scale, test_Vx_low_scale, test_Vy_low_scale, test_Vz_low_scale, test_P_high_scale)



train_dl = DataLoader(train_ds, batch_size=batch_size)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)
test_dl  = DataLoader(test_ds, batch_size=batch_size)

'''
########################
# Unforeseen test data #
########################
# Use concurrent futures to parallelize file reading
with concurrent.futures.ThreadPoolExecutor() as executor:
    for unforeseen_data_for_i in executor.map(load_unforeseen_data_for_index, range(start_index, end_index+1)):
        for p_LR, p_HR, ux_LR, uy_LR, uz_LR, sk_LR in unforeseen_data_for_i:
            unforeseen_matrices_LR.append(p_LR)
            unforeseen_matrices_HR.append(p_HR)
            unforeseen_ux_matrices_LR.append(ux_LR)
            unforeseen_uy_matrices_LR.append(uy_LR)
            unforeseen_uz_matrices_LR.append(uz_LR)
            unforeseen_sk_matrices_LR.append(sk_LR)  

unforeseen_Phigh_test = scaler_Phigh.transform(unforeseen_Phigh_test)
unforeseen_Plow_test  = scaler_Plow.transform(unforeseen_Plow_test)
unforeseen_skull_test = scaler_Slow.transform(unforeseen_skull_test)
unforeseen_Vx_test = scaler_Vxlow.transform(unforeseen_Vx_test)
unforeseen_Vy_test = scaler_Vylow.transform(unforeseen_Vy_test)
unforeseen_Vz_test = scaler_Vzlow.transform(unforeseen_Vz_test)

for i in range(unforeseen_N_test):
    unforeseen_test_P_high_scale[i, 0, :, :, :]     = np.reshape(unforeseen_Phigh_test[i, :], (nx_high, ny_high, nz_high))
    unforeseen_test_P_low_scale [i, 0, :, :, :]     = np.reshape(unforeseen_Plow_test[i, :],   (nx_low,  ny_low,  nz_low))
    unforeseen_test_skull_low_scale[i, 0, :, :, :]  = np.reshape(unforeseen_skull_test[i, :],  (nx_low,  ny_low,  nz_low))
    unforeseen_test_Vx_low_scale [i, 0, :, :, :]    = np.reshape(unforeseen_Vx_test[i, :],     (nx_low,  ny_low,  nz_low))
    unforeseen_test_Vy_low_scale [i, 0, :, :, :]    = np.reshape(unforeseen_Vy_test[i, :],     (nx_low,  ny_low,  nz_low))
    unforeseen_test_Vz_low_scale [i, 0, :, :, :]    = np.reshape(unforeseen_Vz_test[i, :],     (nx_low,  ny_low,  nz_low))

unforeseen_test_ds  = tFUSFormer5chDataset(unforeseen_test_P_low_scale, unforeseen_test_skull_low_scale, unforeseen_test_Vx_low_scale, unforeseen_test_Vy_low_scale, unforeseen_test_Vz_low_scale, unforeseen_test_P_high_scale)
unforeseen_test_dl  = DataLoader(unforeseen_test_ds, batch_size=batch_size)
'''

print('N_train = ', N_train)
print('N_valid = ', N_valid)
print('N_test (foreseen) = ', N_test)
print('N_test (unforeseen) = ', unforeseen_N_test)
