#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:40:14 2019

@author: jonyoung
"""

import numpy as np
import scipy.linalg as la
from sklearn.covariance import GraphLassoCV, GraphLasso
from sklearn.preprocessing import StandardScaler, scale
from sklearn.cross_validation import KFold
from sklearn.covariance.empirical_covariance_ import log_likelihood
import py_quic
from scipy import linalg

def QUIC(data_sample_cov, lambda_val) :
    
    # run QUIC
    try :
        
        X, W, opt, cputime, iters, dGap = py_quic.quic(S=data_sample_cov, L=lambda_val, max_iter=1500, msg=2)
        W = W / (1 + lambda_val)
    
    except :    
        
        print('An error in QUIC calculation occured.')
        
    return X, W, opt, cputime, iters, dGap
    
def QUICCV(data, n_cv_folds) :
    
    # values of regularisation parameter lambda to try
    # 0, 0.05, 0.01,...,0.95
    lambdas = np.arange(20) * 0.05
    
    # set up number of CV folds
    # default is 3 like GraphicalLassoCV
    if n_cv_folds is None :
        
        n_cv_folds = 3
    
    # dimensions of data
    n_parcelations, n_timepoints,  = np.shape(data)   
        
    # store log-likelihood values - 10 folds x 20 lambda values
    log_liks = np.zeros((20,))
       
    # 10-fold cv loop
    kf = KFold(n_timepoints, n_cv_folds)
    
    i = 0
    for train_index, test_index in kf:
        
        print 'i=' + str(i)
        
        # split the data
        training_data = data[:, train_index]
        testing_data = data[:, test_index]
        
        # normalise data with training means and stds
        training_means = np.mean(training_data, axis=1)
        training_stds = np.std(training_data, axis=1)
        training_data_norm = scale(training_data, axis=1)
        testing_data_norm = (testing_data - training_means[:, None]) / training_stds[:, None]      
        
        # calculate sample covariance matrix to initialise QUIC        
        training_sample_cov = np.dot(training_data_norm, np.transpose(training_data_norm)) / n_timepoints
        testing_sample_cov = np.dot(testing_data_norm, np.transpose(testing_data_norm)) / n_timepoints
        
        # loop through lambda values
        for lambda_val, i in zip(lambdas, range(20)) :

            lambda_val = float(lambda_val)
            
            # do the sparse inverse covariance matrix estimation            
            X, cov, opt, cputime, iters, dGap = QUIC(training_sample_cov, lambda_val)
            
            # calculate log-likelihood of the test data under this estimated covariance matrix
            precision = linalg.pinvh(cov)
            log_lik = log_likelihood(testing_sample_cov, precision)
            
            # add log likelihood to the total for this lambda
            log_liks[i] = log_liks[i] + log_lik
            
        print log_liks
        i = i+1
            
    # find lambda value giving best CV log likelihood
    best_lambda_val = float(np.argmax(log_liks) * 0.05)

    # do QUIC on whole data with the best lambda
    data_sample_cov = np.dot(data, np.transpose(data)) / n_timepoints
    X, W, opt, cputime, iters, dGap = QUIC(data_sample_cov, best_lambda_val)
    #W = W / (1 + best_lambda_val)
    return best_lambda_val, W

def get_covariance(data, method, lambda_val='CV', do_scale=False, n_cv_folds=None) :
    
    # default cov if it is not calculated properly
    cov = -1
    
    # scale timecourse
    if do_scale :
        
        data = scale(data, axis=1)
                
    # select method
    if method == 'QUIC' :
        
        # select whether to use supplied regularisation parameter or find the
        # best regularisation parameter by cross validation and maximum likelihood       
        if lambda_val == 'CV' :
            
            best_lambda_val, cov = QUICCV(data, n_cv_folds)
            
        elif isinstance(lambda_val, float) and lambda_val > 0 and lambda_val < 1:
            
             # dimensions of data
            n_parcelations, n_timepoints,  = np.shape(data)   

            # calculate sample covariance matrix
            sample_cov = np.dot(data, np.transpose(data)) / n_timepoints
            
            # do the sparse inverse covariance matrix estimation            
            X, cov, opt, cputime, iters, dGap = QUIC(sample_cov, lambda_val)
        
        else :
            
            print 'lambda_val must be a float between 0 and 1, or "CV" to find the best value by cross-validation'
            
    elif method == 'graphLasso' :
        
                    
        # transpose data as graphLasso likes it this way round
        data = np.transpose(data)
        
        # select whether to use supplied regularisation parameter or find the
        # best regularisation parameter by cross validation and maximum likelihood
        # use scikit-learn implementation of graph lasso and CV graph lasso
        if lambda_val == 'CV' :
            
            try :
                
                model = GraphLassoCV(max_iter=1500, cv=n_cv_folds, assume_centered=True)
                model.fit(data)
                cov = model.covariance_
                
            except :
                
                print('An error in cross validated graphLasso calculation occured.')
             
        elif isinstance(lambda_val, float) and lambda_val > 0 and lambda_val < 1:
            
            try :
            
                model = GraphLasso(alpha=lambda_val, mode='cd', tol=0.0001, max_iter=1500, verbose=False)
                model.fit(data)
                cov = model.covariance_
                
            except FloatingPointError, e:
                
                print('A floating point error in cross validated graphLasso calculation occured.')
                print e
        
        else :
            
            print 'lambda_val must be a float between 0 and 1, or "CV" to find the best value by cross-validation'
        
        
    
    # select method
    else :
        
        print 'Method must be one of "graphLasso" or "QUIC".'
        
    return cov
    
    