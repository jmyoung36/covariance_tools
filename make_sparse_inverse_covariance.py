#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:40:14 2019

@author: jonyoung
"""

import numpy as np
import scipy.linalg as la
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import KFold
from sklearn.covariance.empirical_covariance_ import log_likelihood
from scipy import linalg
#from skggm import QuicGraphicalLasso, QuicGraphicalLassoCV, QuicGraphicalLassoEBIC
#from inverse_covariance import QuicGraphicalLasso, QuicGraphicalLassoCV, QuicGraphicalLassoEBIC

def get_covariance(data, method, lambda_val='CV', do_scale=False, n_cv_folds=None) :
    
    # default cov if it is not calculated properly
    cov = -1
    
    # scale timecourse
    if do_scale :
        
        data = scale(data, axis=1)
                
    # select method
    if method == 'QUIC' :
      
        if lambda_val == 'CV' :
            
            # set up model
            model = QuicGraphicalLassoCV(cv=n_cv_folds)
    
            # fit data to model and return resulting covariance
            model.fit(np.transpose(data))
            return model.covariance_
            
        elif lambda_val == 'EBIC' :
            
            # set up model
            model = QuicGraphicalLassoEBIC()
    
            # fit data to model and return resulting covariance
            model.fit(np.transpose(data))
            return model.covariance_
            
        elif isinstance(lambda_val, float) and lambda_val > 0 and lambda_val < 1:
            
            # set up model
            model = QuicGraphicalLasso(lam=lambda_val)
    
            # fit data to model and return resulting covariance
            model.fit(data)
            return model.covariance_
            
        else :
            
            print ('Error in QUIC covariance:')
            print ('lambda_val must be a float between 0 and 1, "CV" to find the best value by cross-validation, or "EBIC" to use extended Bayesian information criterion for model selection.') 
        
    elif method == 'graphLasso' :
                            
        # transpose data as graphLasso likes it this way round
        data = np.transpose(data)
        
        # select whether to use supplied regularisation parameter or find the
        # best regularisation parameter by cross validation and maximum likelihood
        # use scikit-learn implementation of graph lasso and CV graph lasso
        if lambda_val == 'CV' :
            
            try :
                
                model = GraphicalLassoCV(max_iter=1500, cv=n_cv_folds, assume_centered=True)
                model.fit(data)
                cov = model.covariance_
                
            except :
                
                print('An error in cross validated graphLasso calculation occured.')
             
        elif isinstance(lambda_val, float) and lambda_val > 0 and lambda_val < 1:
            
            try :
            
                model = GraphicalLasso(alpha=lambda_val, mode='cd', tol=0.0001, max_iter=1500, verbose=False)
                model.fit(data)
                cov = model.covariance_
                
            except (FloatingPointError, e):
                
                print('A floating point error in cross validated graphLasso calculation occured.')
                print (e)
        
        else :
            
            print ('Error in graphLasso covariance:')
            print ('lambda_val must be a float between 0 and 1, or "CV" to find the best value by cross-validation')
    
    # select method
    else :
        
        print ('Method must be one of "graphLasso" or "QUIC".')
        
    return cov
    
    