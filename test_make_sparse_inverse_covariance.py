#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:11:14 2019

@author: jonyoung
"""

import numpy as np
from make_sparse_inverse_covariance import get_covariance
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler, scale
from scipy.linalg import logm

# set directories
data_dir = '/home/jonyoung/IoP_data/Data/PSYSCAN/Legacy_data/Dublin/Dublin_fMRI_forJ/'

# read in the .mat file containing the time series
timeseries_data = loadmat(data_dir + 'wavelets2_save.mat')['wavelets_save2']
timeseries = timeseries_data[0][0]
timeseries = np.transpose(timeseries)
timeseries = timeseries[:60, :40]

cov_lasso_CV = get_covariance(timeseries, 'graphLasso', lambda_val = 'CV', do_scale=True)
#print np.shape(cov)
#print cov[:4, :4]
#print logm(cov)[:4, :4]
cov_QUIC_CV = get_covariance(timeseries, 'QUIC', lambda_val = 'CV', do_scale=True)
cov_QUIC_EBIC = get_covariance(timeseries, 'QUIC', lambda_val = 'EBIC', do_scale=True)