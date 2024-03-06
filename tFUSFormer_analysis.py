# All trained models used in our experiments, including those for the super-resolution task, segmentation task,
# gene expression prediction task, survival prediction task, and classification task, have been uploaded to
# the provided data drive link (MDA_Datasets/data/Trained Models/...).
# Here, we provide an example of the feature extraction process from Tensorflow models for the segmentation task.
# For PyTorch models, the feature extraction process is demonstrated in example_pytorch_feature_extraction.py.

from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf

# Path to the pre-trained model
model_path='../tFUSFormer/model/model_FSRCNN_1ch.pth'
# Load the pre-trained model
model=load_model(model_path)

# Load test data
X_test=np.load('../tFUSFormer/test_results/model_FSRCNN_1ch_seen/SR.npy')

# Extract output from a specific layer ('conv2d_9') of the model
interlayer_output=model.get_layer('mid_part').output
# Create a new model that outputs the interlayer output
inter_model = tf.keras.Model(inputs=model.input, outputs=interlayer_output)

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
