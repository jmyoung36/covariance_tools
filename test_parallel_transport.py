#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:18:45 2019

@author: jonyoung
"""

import numpy as np
from sklearn import svm, cross_validation, metrics, model_selection
from SPD_manifold_tools import geometric_mean, transport_schilds_ladder, transport_yair
import pandas as pd
import csv
import scipy.linalg as la
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import connectivity_utils as utils

        
# set directories
dataset_1_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC_Unsmooth_TimeCourse/'
dataset_3_dir = '/home/jonyoung/IoP_data/Data/connectivity_data/KCL3_timecourse/'

# indices of lower triangular elements
lotril_inds = np.array([np.ravel_multi_index(np.tril_indices(90, k=-1), (90, 90))])

# import and process sparse inverse covariance data and metadata from dataset 1
# import dataset 1 sparse inverse covariances
dataset_1_cov_data = np.loadtxt(dataset_1_dir + 'sparse_inverse_covariance_data.csv', delimiter=',')

# import dataset 1 sparse inverse covariance files 
dataset_1_cov_files = pd.read_csv(dataset_1_dir + 'sparse_inverse_covariance_files.csv').T

# put these in a df
dataset_1_cov = pd.DataFrame(data=dataset_1_cov_data)
dataset_1_cov['file'] = dataset_1_cov_files.index

# convert format of file name so they can be matched
dataset_1_cov['file'] = dataset_1_cov['file'].apply(lambda x: x.split('/')[-1].split('_')[-1].zfill(7))

# import and process full dataset 1 files list and metadata to get labels
# import original dataset 1 data, files
dataset_1_data, dataset_1_files = utils.load_connectivity_data('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/')

# import dataset 1 labels
dataset_1_labels = utils.load_labels('/home/jonyoung/IoP_data/Data/connectivity_data/KCL_SC1/matrix_unsmooth/')

# put labels alongside files in a DF
dataset_1_metadata = pd.DataFrame(columns=['file', 'label'])
dataset_1_metadata['file'] = dataset_1_files
dataset_1_metadata['label'] = dataset_1_labels 

# convert format of file name so they can be matched
dataset_1_metadata['file'] = dataset_1_metadata['file'].apply(lambda x: x.split('/')[-1].split('_')[-1])

# join the DFs to match labels with spare inverse cov data
dataset_1_cov = dataset_1_cov.merge(dataset_1_metadata, how='inner', on='file')

# extract the data and labels
dataset_1_cov_data = dataset_1_cov.iloc[:,0:8100].as_matrix()
dataset_1_cov_labels = np.array(dataset_1_cov['label'].tolist())

# shuffle the rows and labels
r = np.random.permutation(len(dataset_1_cov_labels))
dataset_1_cov_data_s = dataset_1_cov_data[r, :]
#dataset_1_cov_data_s = np.transpose(dataset_1_cov_data_s)
dataset_1_cov_labels_s = dataset_1_cov_labels[r]

# import and process sparse inverse covariance data and metadata from dataset 3
# import dataset 1 sparse inverse covariances and files
dataset_3_cov_data = np.loadtxt(dataset_3_dir + 'sparse_inverse_covariance_data.txt', delimiter=',')
with open(dataset_3_dir + 'sparse_inverse_covariance_files.csv', 'rb') as f:
    reader = csv.reader(f)
    dataset_3_cov_files = list(reader)[0]

# generate list of labels from file names: 1=pat (patient), 0=con (control)
dataset_3_cov_labels = np.array([1 if 'pat' in filename else 0 for filename in dataset_3_cov_files])

# shuffle the rows and labels
r = np.random.permutation(len(dataset_3_cov_labels))
dataset_3_cov_data_s = dataset_3_cov_data[r, :]
#dataset_3_cov_data_s = np.transpose(dataset_3_cov_data_s)
dataset_3_cov_labels_s = dataset_3_cov_labels[r]

# calculate the (geometric) means
# reshape
dataset_1_cov_data_s = np.reshape(dataset_1_cov_data_s, (100, 90, 90))
dataset_3_cov_data_s = np.reshape(dataset_3_cov_data_s, (150, 90, 90))

# get means
dataset_1_geo_mean = geometric_mean(dataset_1_cov_data_s)
dataset_3_geo_mean = geometric_mean(dataset_3_cov_data_s)

# parallel transport dataset 1 onto dataset 3 with Schild's ladder
schild_parallel_transported_dataset_1 = transport_schilds_ladder(dataset_1_geo_mean, dataset_3_geo_mean, dataset_1_cov_data_s, n_steps=25)
schild_parallel_transported_dataset_1_mean = geometric_mean(schild_parallel_transported_dataset_1)

# parallel transport dataset 1 onto dataset 3 with Yair's method
yair_parallel_transported_dataset_1 = transport_yair(dataset_1_geo_mean, dataset_3_geo_mean, dataset_1_cov_data_s)
yair_parallel_transported_dataset_1_mean = geometric_mean(yair_parallel_transported_dataset_1)


# try Euclidean mean shift
mean_shift_vector = dataset_3_geo_mean - dataset_1_geo_mean
Euclidean_transported_dataset_1 = np.zeros_like(dataset_1_cov_data_s)
for i in range(np.shape(dataset_1_cov_data_s)[0]) :
    Euclidean_transported_dataset_1[i, :, :] = dataset_1_cov_data_s[i, :, :] + mean_shift_vector
    
# combine all the datasets and extract the lower triangle to eliminate redundant components
combined_untransported_dataset = np.reshape(np.vstack((dataset_1_cov_data_s, dataset_3_cov_data_s)), (250, 8100))[:, lotril_inds[0]]
combined_schilds_ladder_dataset = np.reshape(np.vstack((schild_parallel_transported_dataset_1, dataset_3_cov_data_s)), (250, 8100))[:, lotril_inds[0]]
combined_yair_dataset = np.reshape(np.vstack((yair_parallel_transported_dataset_1, dataset_3_cov_data_s)), (250, 8100))[:, lotril_inds[0]]
combined_mean_shift_dataset = np.reshape(np.vstack((Euclidean_transported_dataset_1, dataset_3_cov_data_s)), (250, 8100))[:, lotril_inds[0]]
  
# do joint PCA on the combined data
pca = PCA(n_components = 2)
combined_untransported_dataset_PCA = pca.fit_transform(combined_untransported_dataset)
combined_schilds_ladder_dataset_PCA = pca.fit_transform(combined_schilds_ladder_dataset)
combined_yair_dataset_PCA = pca.fit_transform(combined_yair_dataset)
combined_mean_shift_dataset_PCA = pca.fit_transform(combined_mean_shift_dataset)

plt.figure()
plt.scatter(combined_untransported_dataset_PCA[1:100,0],combined_untransported_dataset_PCA[1:100 ,1], color='blue')
plt.scatter(combined_untransported_dataset_PCA[100:,0],combined_untransported_dataset_PCA[100: ,1], color='red')
plt.title('PCA of datasets 1 and 3, no transport')

plt.figure()
plt.scatter(combined_schilds_ladder_dataset_PCA[1:100,0],combined_schilds_ladder_dataset_PCA[1:100 ,1], color='blue')
plt.scatter(combined_schilds_ladder_dataset_PCA[100:,0],combined_schilds_ladder_dataset_PCA[100: ,1], color='red')
plt.title("PCA of datasets 1 and 3, transport with Schild's ladder")

plt.figure()
plt.scatter(combined_yair_dataset_PCA[1:100,0],combined_yair_dataset_PCA[1:100 ,1], color='blue')
plt.scatter(combined_yair_dataset_PCA[100:,0],combined_yair_dataset_PCA[100: ,1], color='red')
plt.title("PCA of datasets 1 and 3, transport with Yair's method")

plt.figure()
plt.scatter(combined_mean_shift_dataset_PCA[1:100,0],combined_mean_shift_dataset_PCA[1:100 ,1], color='blue')
plt.scatter(combined_mean_shift_dataset_PCA[100:,0],combined_mean_shift_dataset_PCA[100: ,1], color='red')
plt.title("PCA of datasets 1 and 3, transport with Euclidean mean shift")