import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
import concurrent.futures
import scipy 
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from config import dir_data, test_data_mode
import h5py

# Use the configurations
batch_size = config.batch_size

class tFUSDataset(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as file:
            self.length = len(file.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with h5py.File(self.file_path, 'r') as file:
            group = file[str(index)]
            sim_data1 = torch.tensor(group['sim_data1'][:], dtype=torch.float32)
            sim_data2 = torch.tensor(group['sim_data2'][:], dtype=torch.float32)
            sim_data3 = torch.tensor(group['sim_data3'][:], dtype=torch.float32)
            sim_data4 = torch.tensor(group['sim_data4'][:], dtype=torch.float32)
            sim_data5 = torch.tensor(group['sim_data5'][:], dtype=torch.float32)
            label = torch.tensor(group['label'][:], dtype=torch.float32)
        return sim_data1, sim_data2, sim_data3, sim_data4, sim_data5, label

# Assuming 'train_ds.hdf5' is your file path
train_ds = tFUSDataset(f'{dir_data}/train_ds.hdf5')
valid_ds = tFUSDataset(f'{dir_data}/valid_ds.hdf5')
test_ds  = tFUSDataset(f'{dir_data}/foreseen_test_ds.hdf5')
unforeseen_test_ds = []
if test_data_mode == 'seen':
    print("test_data_mode = seen")
elif test_data_mode == 'unseen1':
    unforeseen_test_ds = tFUSDataset(f'{dir_data}/unforeseen_test_sk13_ds.hdf5')
elif test_data_mode == 'unseen2':
    unforeseen_test_ds = tFUSDataset(f'{dir_data}/unforeseen_test_sk17_ds.hdf5')
else:
    print("Invalid test_data_mode value or dataset not found.")

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
unforeseen_test_dl = DataLoader(unforeseen_test_ds, batch_size=batch_size)
