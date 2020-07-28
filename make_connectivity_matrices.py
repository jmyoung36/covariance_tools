#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:29:35 2019

@author: jonyoung
"""

# import what we need
import numpy as np
from scipy.io import loadmat
import sys
sys.path.append("/home/k1511004/Projects/covariance_tools/") 
from make_sparse_inverse_covariance import get_covariance

# set directories
data_dir = '/home/k1511004/Data/PSYSCAN/WP5_data/Prelim_FEP_dataset/253_subjects_Apr_20/Data/fMRI/'

# read in the .mat file containing the time series
timeseries_data = loadmat(data_dir + 'wavelets2_save.mat')['wavelets2_save'][0]
# loop through the subjects
n_subjects = len(timeseries_data)
n_regions = np.shape(timeseries_data[0])[1]
n_regions_cortical = n_regions - 15
covariance_data = np.zeros((n_subjects, n_regions_cortical * n_regions_cortical))
covariance_data_scaled = np.zeros((n_subjects, n_regions_cortical * n_regions_cortical))
for i in range(n_subjects) :
#for i in range(3) :
    
    print (i)
    subject_timeseries_data = np.transpose(timeseries_data[i])
    
    # remove first 16 ROIs - subcortical
    subject_timeseries_data = subject_timeseries_data[15:, :]
#    
#    try :
#    
#        subject_covariance_data = get_covariance(subject_timeseries_data, method = 'graphLasso', lambda_val='CV', do_scale=False, n_cv_folds=None)
#        print (np.shape(subject_covariance_data))
#        print (subject_covariance_data[:5, :5])
#        covariance_data[i, :] = np.reshape(subject_covariance_data, (1, n_regions_cortical * n_regions_cortical))
#        
#    
#    except (TypeError, ValueError) as e :
#        
#        print ('Covariance matrix estimation failed for subject ' + str(i))
#        print (e)
    
    try :
        print (np.shape(subject_timeseries_data))
        subject_covariance_data_scaled = get_covariance(subject_timeseries_data, method = 'graphLasso', lambda_val='CV', do_scale=True, n_cv_folds=None)
        print (np.shape(subject_covariance_data_scaled))
        print (subject_covariance_data_scaled[:5, :5])
        covariance_data_scaled[i, :] = np.reshape(subject_covariance_data_scaled, (1, n_regions_cortical * n_regions_cortical))
        
    except (TypeError, ValueError) as e :
        
        print ('Scaled covariance matrix estimation failed for subject ' + str(i))
        print (e)
        
# save the connectivity data
#np.save(data_dir + 'covariance_data_QUIC', covariance_data)
np.save(data_dir + 'covariance_data_scaled_GraphLasso', covariance_data_scaled)