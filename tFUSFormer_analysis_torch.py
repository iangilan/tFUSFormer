import torch
import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms
from models import tFUSFormer_1ch

# Assuming model_path points to a .pth file or similar PyTorch format
model_path='../tFUSFormer/model/model_tFUSFormer_1ch.pth'
model = tFUSFormer_1ch()  # You need to define your model architecture
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Load test data
X_test=np.load('../tFUSFormer/test_results/model_tFUSFormer_1ch_seen/SR.npy')

# Normalize test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std according to your dataset
])
X_test = torch.stack([transform(x) for x in X_test])

def get_intermediate_layer(module, input, output):
    global inter_out
    inter_out.append(output)

# Assuming 'conv2d_9' is the name of the layer you're interested in
layer_name = 'RSTB'
hook = getattr(model, layer_name).register_forward_hook(get_intermediate_layer)

# Initialize list to store outputs
inter_out = []

for i in range(len(X_test)):
    test_img = X_test[i].unsqueeze(0)  # Add batch dimension
    model(test_img)  # Forward pass (outputs captured by the hook)

# Detach and convert to NumPy
inter_out = [o.detach().numpy() for o in inter_out]
inter_out = np.array(inter_out)

# Reshape for PCA as before
n1, h1, w1, c1 = inter_out.shape
inter_out = inter_out.reshape(-1, h1*w1*c1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=400, svd_solver='arpack')
inter_out_pca = pca.fit_transform(inter_out)
print(inter_out_pca.shape)

# Save the PCA-transformed features
np.save('../data/Seg/feature_test.npy', inter_out_pca)

hook.remove()











# Initialize an empty list to store outputs
inter_out=[]
# Loop through the test data to extract features
for i in range(len(X_test)):
    test_img=X_test[i]  # Get an individual test image
    test_img=test_img[np.newaxis,:, :]  # Add an extra dimension
    test_img=test_img/255  # Normalize the image
    test_out=inter_model.predict(test_img)  # Predict using the intermediate model
    test_out=np.squeeze(test_out)  # Remove single-dimensional entries
    inter_out.append(test_out)  # Append the output to the list
# Convert list to numpy array
inter_out=np.array(inter_out)

# Reshape the output for PCA
n1, h1, w1, c1 = inter_out.shape
inter_out = inter_out.reshape(-1, h1*w1*c1)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=400, svd_solver='arpack')
inter_out = pca.fit_transform(inter_out)
print(inter_out.shape)

# Save the PCA-transformed features
np.save('../mda/results/FSRCNN1ch_feature_test.npy',inter_out)

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
testDataFeatures = np.load('../data/SR/feature4_test_pca.npy')
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
