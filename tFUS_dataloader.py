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
end_index = 600


# Initialize lists to store the matrices
lr_data, hr_data, Vx_data, Vy_data, Vz_data, sk_data = [], [], [], [], [], []

with concurrent.futures.ThreadPoolExecutor() as executor:
    data_for_all_indices = executor.map(load_data_for_index, range(start_index, end_index + 1))
    for data_for_i in data_for_all_indices:
        lr_data.extend([item[0] for item in data_for_i])
        hr_data.extend([item[1] for item in data_for_i])
        Vx_data.extend([item[2] for item in data_for_i])
        Vy_data.extend([item[3] for item in data_for_i])
        Vz_data.extend([item[4] for item in data_for_i])
        sk_data.extend([item[5] for item in data_for_i])

# Calculate start and end indexes for extraction
start_index1, end_index1 = int((32 - 25) / 2), int((32 + 25) / 2)
start_index2, end_index2 = int((128 - 100) / 2), int((128 + 100) / 2)

# Convert lists to numpy arrays for efficient slicing
lr_data = np.array(lr_data)
hr_data = np.array(hr_data)
Vx_data = np.array(Vx_data)
Vy_data = np.array(Vy_data)
Vz_data = np.array(Vz_data)
sk_data = np.array(sk_data)

# Extract the center portions using advanced slicing
lr_data = lr_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
Vx_data = Vx_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
Vy_data = Vy_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
Vz_data = Vz_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
sk_data = sk_data[:, :25, :25, :25]  # Assuming the shape is already 25x25x25
hr_data = hr_data[:, start_index2:end_index2, start_index2:end_index2, start_index2:end_index2]


N_samples_hr, nx_high, ny_high, nz_high = hr_data.shape
N_samples_lr, nx_low, ny_low, nz_low    = lr_data.shape
nxyz_high = nz_high*nx_high*ny_high
nxyz_low  = nz_low*nx_low*ny_low
N = N_samples_lr

hr_data = hr_data.reshape(N, nxyz_high)
lr_data = lr_data.reshape(N, nxyz_low)
sk_data = sk_data.reshape(N, nxyz_low)
Vx_data = Vx_data.reshape(N, nxyz_low)
Vy_data = Vy_data.reshape(N, nxyz_low)
Vz_data = Vz_data.reshape(N, nxyz_low)


Plow_train, Plow_valid, Phigh_train, Phigh_valid = train_test_split(lr_data, hr_data, test_size = 1200, random_state = 1)
del hr_data, lr_data
skull_train, skull_valid = train_test_split(sk_data, test_size = 1200, random_state = 1)
del sk_data
Vx_train, Vx_valid = train_test_split(Vx_data, test_size = 1200, random_state = 1)
del Vx_data 
Vy_train, Vy_valid = train_test_split(Vy_data, test_size = 1200, random_state = 1)
del Vy_data
Vz_train, Vz_valid = train_test_split(Vz_data, test_size = 1200, random_state = 1)
del Vz_data

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

# Reshape the training data
Phigh_train  = Phigh_train.reshape(N_train, 1, nx_high, ny_high, nz_high)
Plow_train   = Plow_train.reshape(N_train, 1, nx_low, ny_low, nz_low)
skull_train  = skull_train.reshape(N_train, 1, nx_low, ny_low, nz_low)
Vx_train     = Vx_train.reshape(N_train, 1, nx_low, ny_low, nz_low)
Vy_train     = Vy_train.reshape(N_train, 1, nx_low, ny_low, nz_low)
Vz_train     = Vz_train.reshape(N_train, 1, nx_low, ny_low, nz_low)

# Reshape the validation data
Phigh_valid  = Phigh_valid.reshape(N_valid, 1, nx_high, ny_high, nz_high)
Plow_valid   = Plow_valid.reshape(N_valid, 1, nx_low, ny_low, nz_low)
skull_valid  = skull_valid.reshape(N_valid, 1, nx_low, ny_low, nz_low)
Vx_valid     = Vx_valid.reshape(N_valid, 1, nx_low, ny_low, nz_low)
Vy_valid     = Vy_valid.reshape(N_valid, 1, nx_low, ny_low, nz_low)
Vz_valid     = Vz_valid.reshape(N_valid, 1, nx_low, ny_low, nz_low)

# Reshape the test data
Phigh_test  = Phigh_test.reshape(N_test, 1, nx_high, ny_high, nz_high)
Plow_test   = Plow_test.reshape(N_test, 1, nx_low, ny_low, nz_low)
skull_test  = skull_test.reshape(N_test, 1, nx_low, ny_low, nz_low)
Vx_test     = Vx_test.reshape(N_test, 1, nx_low, ny_low, nz_low)
Vy_test     = Vy_test.reshape(N_test, 1, nx_low, ny_low, nz_low)
Vz_test     = Vz_test.reshape(N_test, 1, nx_low, ny_low, nz_low)


#=========================================================================================================
class tFUS_Dataset(Dataset):
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

train_ds = tFUS_Dataset(Plow_train, skull_train, Vx_train, Vy_train, Vz_train, Phigh_train)
valid_ds = tFUS_Dataset(Plow_valid, skull_valid, Vx_valid, Vy_valid, Vz_valid, Phigh_valid)
test_ds  = tFUS_Dataset(Plow_test, skull_test, Vx_test, Vy_test, Vz_test, Phigh_test)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


########################
# Unforeseen test data #
########################

# Define the range of file numbers to load
start_index = 1
end_index = 200

# Initialize lists to store the matrices
unforeseen_lr_data, unforeseen_hr_data, unforeseen_Vx_data, unforeseen_Vy_data, unforeseen_Vz_data, unforeseen_sk_data = [], [], [], [], [], []

with concurrent.futures.ThreadPoolExecutor() as executor:
    unforeseen_data_for_all_indices = executor.map(load_unforeseen_data_for_index, range(start_index, end_index + 1))
    for unforeseen_data_for_i in unforeseen_data_for_all_indices:
        unforeseen_lr_data.extend([item[0] for item in unforeseen_data_for_i])
        unforeseen_hr_data.extend([item[1] for item in unforeseen_data_for_i])
        unforeseen_Vx_data.extend([item[2] for item in unforeseen_data_for_i])
        unforeseen_Vy_data.extend([item[3] for item in unforeseen_data_for_i])
        unforeseen_Vz_data.extend([item[4] for item in unforeseen_data_for_i])
        unforeseen_sk_data.extend([item[5] for item in unforeseen_data_for_i])


# Convert lists to numpy arrays for efficient slicing
unforeseen_lr_data = np.array(unforeseen_lr_data)
unforeseen_hr_data = np.array(unforeseen_hr_data)
unforeseen_Vx_data = np.array(unforeseen_Vx_data)
unforeseen_Vy_data = np.array(unforeseen_Vy_data)
unforeseen_Vz_data = np.array(unforeseen_Vz_data)
unforeseen_sk_data = np.array(unforeseen_sk_data)

# Extract the center portions using advanced slicing
unforeseen_lr_data = unforeseen_lr_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
unforeseen_Vx_data = unforeseen_Vx_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
unforeseen_Vy_data = unforeseen_Vy_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
unforeseen_Vz_data = unforeseen_Vz_data[:, start_index1:end_index1, start_index1:end_index1, start_index1:end_index1]
unforeseen_sk_data = unforeseen_sk_data[:, :25, :25, :25]  # Assuming the shape is already 25x25x25
unforeseen_hr_data = unforeseen_hr_data[:, start_index2:end_index2, start_index2:end_index2, start_index2:end_index2]


N_samples_hr, nx_high, ny_high, nz_high = unforeseen_hr_data.shape
N_samples_lr, nx_low, ny_low, nz_low    = unforeseen_lr_data.shape
nxyz_high = nz_high*nx_high*ny_high
nxyz_low  = nz_low*nx_low*ny_low
N = N_samples_lr

unforeseen_hr_data = unforeseen_hr_data.reshape(N, nxyz_high)
unforeseen_lr_data = unforeseen_lr_data.reshape(N, nxyz_low)
unforeseen_sk_data = unforeseen_sk_data.reshape(N, nxyz_low)
unforeseen_Vx_data = unforeseen_Vx_data.reshape(N, nxyz_low)
unforeseen_Vy_data = unforeseen_Vy_data.reshape(N, nxyz_low)
unforeseen_Vz_data = unforeseen_Vz_data.reshape(N, nxyz_low)

unforeseen_N_test  = (unforeseen_lr_data.shape)[0]

## standardize the values of training and testing data from 0 to 1
unforeseen_Phigh_test = scaler_Phigh.transform(unforeseen_hr_data)
del unforeseen_hr_data
unforeseen_Plow_test  = scaler_Plow.transform(unforeseen_lr_data)
del unforeseen_lr_data
unforeseen_skull_test = scaler_Slow.transform(unforeseen_sk_data)
del unforeseen_sk_data
unforeseen_Vx_test    = scaler_Vxlow.transform(unforeseen_Vx_data)
del unforeseen_Vx_data
unforeseen_Vy_test    = scaler_Vylow.transform(unforeseen_Vy_data)
del unforeseen_Vy_data
unforeseen_Vz_test    = scaler_Vzlow.transform(unforeseen_Vz_data)
del unforeseen_Vz_data

# Reshape the test data
unforeseen_Phigh_test = unforeseen_Phigh_test.reshape(unforeseen_N_test, 1, nx_high, ny_high, nz_high)
unforeseen_Plow_test  = unforeseen_Plow_test.reshape(unforeseen_N_test, 1, nx_low, ny_low, nz_low)
unforeseen_skull_test = unforeseen_skull_test.reshape(unforeseen_N_test, 1, nx_low, ny_low, nz_low)
unforeseen_Vx_test    = unforeseen_Vx_test.reshape(unforeseen_N_test, 1, nx_low, ny_low, nz_low)
unforeseen_Vy_test    = unforeseen_Vy_test.reshape(unforeseen_N_test, 1, nx_low, ny_low, nz_low)
unforeseen_Vz_test    = unforeseen_Vz_test.reshape(unforeseen_N_test, 1, nx_low, ny_low, nz_low)

for i in range(unforeseen_N_test):
    unforeseen_Phigh_test[i, 0, :, :, :] = np.reshape(unforeseen_Phigh_test[i, :], (nx_high, ny_high, nz_high))
    unforeseen_Plow_test[i, 0, :, :, :]  = np.reshape(unforeseen_Plow_test[i, :],   (nx_low,  ny_low,  nz_low))
    unforeseen_skull_test[i, 0, :, :, :] = np.reshape(unforeseen_skull_test[i, :],  (nx_low,  ny_low,  nz_low))
    unforeseen_Vx_test[i, 0, :, :, :]    = np.reshape(unforeseen_Vx_test[i, :],     (nx_low,  ny_low,  nz_low))
    unforeseen_Vy_test[i, 0, :, :, :]    = np.reshape(unforeseen_Vy_test[i, :],     (nx_low,  ny_low,  nz_low))
    unforeseen_Vz_test[i, 0, :, :, :]    = np.reshape(unforeseen_Vz_test[i, :],     (nx_low,  ny_low,  nz_low))

unforeseen_test_ds = tFUS_Dataset(unforeseen_Plow_test, unforeseen_skull_test, unforeseen_Vx_test, unforeseen_Vy_test, unforeseen_Vz_test, unforeseen_Phigh_test)
unforeseen_test_dl = DataLoader(unforeseen_test_ds, batch_size=batch_size)

print('N_train = ', N_train)
print('N_valid = ', N_valid)
print('N_test (foreseen)   = ', N_test)
print('N_test (unforeseen) = ', unforeseen_N_test)
'''
def save_dataloader_to_hdf5(dataloader, hdf5_filename):
    # Initialize lists to store data and labels
    all_data = []
    all_labels = []

    # Iterate through the DataLoader
    for data, labels in dataloader:
        # Assuming data and labels are numpy arrays or can be converted to numpy arrays
        all_data.append(data.numpy())
        all_labels.append(labels.numpy())

    # Concatenate all data and labels
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Save to HDF5
    with h5py.File(hdf5_filename, 'w') as hf:
        hf.create_dataset('data', data=all_data)
        hf.create_dataset('labels', data=all_labels)

# Example usage
# Replace 'train_dl', 'valid_dl', 'test_dl', 'unforeseen_test_dl' with your DataLoader variables
save_dataloader_to_hdf5(train_dl, 'train_data.hdf5')
save_dataloader_to_hdf5(valid_dl, 'valid_data.hdf5')
save_dataloader_to_hdf5(test_dl, 'foreseen_test_data.hdf5')
save_dataloader_to_hdf5(unforeseen_test_dl, 'unforeseen_test_data_sk13.hdf5')
'''
