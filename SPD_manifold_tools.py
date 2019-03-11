# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:25:59 2017

@author: jonyoung

The algorithms in this script are based on the following references:
[1] 'Transport on Riemannian Manifold for Connectivity-Based Brain Decoding', Ng et al, Transactions on Medical Imaging
[2] 'A Plug & Play P300 BCI Using Information Geometry', Barachant et al, arXiv
[3] 'Riemannian Geometry Applied to BCI Classification', Barachant et al, LVA/ICA 2010: Latent Variable Analysis and Signal Separation
[4] 'Parallel Transport on the Cone Manifold of SPD Matrices for Domain Adaptation', Yair et al, arXiv
[5] 'A Riemannian  framework  for tensor  computing', Pennec et al, International  Journal  of  Computer  Vision
[6] 'Log‐Euclidean metrics for fast and simple calculus on diffusion tensors', Arsigny et al., Magnetic Resonance in Medicine

"""

import numpy as np
import scipy.linalg as la

# Generate the tangent vector at M2 'pointing toward' M1, where M1 and M2 are both
# symmetric positive definite matrices in S++. This is the inverse operation of the Exp map.
# From [1]
def log_map(M1, M2) :
    
    M2_pos = la.fractional_matrix_power(M2, 0.5)
    M2_neg = la.fractional_matrix_power(M2, -0.5)
    #return np.linalg.multi_dot([B_pos, la.logm(np.linalg.multi_dot([B_neg, A, B_neg])), B_pos])
    log_prod = la.logm(M2_neg.dot(M1).dot(M2_neg))
    return M2_pos.dot(log_prod).dot(M2_pos)
    
# Project the tangent vector at M2 'pointing toward' M1 back into S++, where M1 
# and M2 are both symmetric positive definite matrices. This is the inverse operation of the Log map.
# From [1]
def exp_map(M1, M2) :
    
    M2_pos = la.fractional_matrix_power(M2, 0.5)
    M2_neg = la.fractional_matrix_power(M2, -0.5)
    #return np.linalg.multi_dot([B_pos, la.expm(np.linalg.multi_dot([B_neg, A, B_neg])), B_pos])
    exp_prod = la.expm(M2_neg.dot(M1).dot(M2_neg))
    return M2_pos.dot(exp_prod).dot(M2_pos)

# create a geodesic, i.e. local shortest path, from matrix M2 to matrix M1 where 
# M1 and M2 are both positive definite matrices on S++. Take an optional distance argument 
# d allowing us to travel e.g. half way from M2 to M1, or twice the distance from
# M2 to M1.
# From [1]
def make_geodesic(M1, M2, d=1):
    
    return exp_map(d * log_map(M1, M2), M2)

# calculate the geometric mean of a set of covariance matrices and use this to project
# the matrices into the tangent space of the mean matrix
def project_to_mean_tangent(matrices) :
    
    # find geometric mean
    # construct base covariance matrix by repeated averaging in tangent space
    # first, initialise base covariance matrix
    base_cov_matrix = np.mean(matrices, axis=0)
    
    for i in range(2) :
        
        #print base_cov_matrix[:5, :5]
    
        # project all matrices into the tangent space
        tangent_matrices = np.zeros_like(matrices)
        for j in range(len(matrices)) :
        
            tangent_matrices[j, :, :] = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.logm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), matrices[j, :, :],la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
    
        # calculate the tangent space mean
        tangent_space_base_cov_matrix = np.mean(tangent_matrices, axis=0)
        
        #print tangent_space_base_cov_matrix[:5, :5]
        
        # project new tangent mean back to the manifold
        base_cov_matrix = np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, 0.5), la.expm(np.linalg.multi_dot([la.fractional_matrix_power(base_cov_matrix, -0.5), tangent_space_base_cov_matrix ,la.fractional_matrix_power(base_cov_matrix, -0.5)])), la.fractional_matrix_power(base_cov_matrix, 0.5)])
        
    # apply whitening transport and projection for training AND testing data
#    projected_matrices = np.zeros_like(matrices)   
#    base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix, -0.5)
#    
#    for i in range(len(matrices)) :
#        projected_matrices[i, :, :] = la.logm(np.linalg.multi_dot([base_cov_matrix_pow, matrices[i, :, :], base_cov_matrix_pow]))
#       
#    return projected_matrices
    return base_cov_matrix

# calculate the Log-Euclidean mean of a set of n dxd matrices in S++, by applying
# a matrix logarithm to project the matrices onto the tangent space at I,
# calculating the arithmetic mean, and then projecting it back into S++.
# Take a stack of n dxd matrices MM, as a numpy array with dimensions d x d x n
# From [1]
def log_Euclidean_mean(MM):
    
    # create a structure to store the tangent space matrices
    logm_MM = np.zeros_like(MM)
    
    # project all the matrices in MM into the tangent space
    for i in range(np.shape(MM)[2]) :
        
        logm_MM[:, :, i] = np.logm(MM[:, :, i])
        
    # take the mean and project back to S++
    mean_logm_MM = np.mean(logm_MM, axis=2)
    return la.expm(mean_logm_MM)

# calculate the geometric mean of a set of n dxd matrices in S++. In the 
# general case, initially calculate the Euclidean mean. Then project the
# matrices into the tangent space of the Euclidean mean with log_map, calculate
# the mean of the projected matrices, and project this back into S++ with 
# exp_map to provide an updated estimate of the geometric mean. Repeat with 
# this new estimate until the estimate converges or the maximum number of 
# iterations is reached. In the special case of two matrices, the geometric 
# mean is simply the midpoint of the geodesic joining them.
# Take a stack of n dxd matrices MM, as a numpy array with dimensions n x d x d
# From [3, 4] for general case and 2 for matrix case respectively 
def geometric_mean(MM, tol=10e-10, max_iter=50):
    
    # number of matrices
    n = np.shape(MM)[0]
    if n == 2: 
        
        # special case - just take midpoint of geodesic joining the matrices [5]
        M1 = np.matrix(np.squeeze(MM[0, :, :]))
        M2 = np.matrix(np.squeeze(MM[1, :, :]))
        M_mean = M1 ** 0.5 * ((M1 ** -0.5 * M2 ** M1 **-0.5) ** 0.5) * M1 **0.5
        return M_mean
    
    else:
        
        # general case, more than 2 matrices
       
        # initialise variables
        # Euclidean mean as first estimate
        new_est = np.mean(MM, axis=0)
        # convergence criterion
        crit = np.finfo(np.float64).max
        k = 0

    
        # start the loop
        while (crit > tol) and (k < max_iter):
        
        
            #print new_est[:5, :5]
        
            # update the current estimate
            current_est = new_est
        
            # project all the matrices into the tangent space of the current estimate
            tan_MM = np.zeros_like(MM)
            for i in range(n) :
            
                #tan_MM[i, :, :] = log_map(current_est, MM[i, :, :])
                tan_MM[i, :, :] = log_map(MM[i, :, :], current_est)
            
            # arithmetic mean in the tangent spacegeometric_mean
            S = np.mean(tan_MM, axis=0)
        
            #print S[:5, :5]
        
            # project S back to S++ to create a new estimated mean
            new_est = exp_map(S, current_est)
            #new_est = exp_map(current_est, S)
        
            # housekeeping: update k and crit
            k = k + 1
            #crit = np.linalg.norm(S, ord='fro')
            crit = np.linalg.norm(new_est - current_est, ord='fro')
            print k
            print crit
        
    return new_est

# take a source and a target mean and an SPD connectivity matrix. Generate a 
# discretised geodesic from the source mean to the target mean. Use this 
# geodesic to parallel transport the connectivity matrices in the 
# source dataset to the location of the target dataset with Schild's Ladder.
# algorith takem from [1]
def Schilds_ladder(source_mean, target_mean, matrices, n_steps=10) :
    
    # generate a discretised geodesic with n_steps step, at source_mean and 
    # pointing toward target_mean
    n_matrices, n_regions, n_regions = np.shape(matrices)
    disc_geo = np.zeros((n_steps, n_regions, n_regions))
    for i in range(n_steps) :
        
        frac_dist = (i+1)/float(n_steps)
        disc_geo[i, :, :] = make_geodesic(target_mean, source_mean, frac_dist)
        
    # initialise the transported matrices
    # copy original matrices BY VALUE 
    transported_source_dataset = np.copy(matrices)
        
    # perform the transport
    print "Performing parallel transport with Schild's ladder..."
    # first n_steps - 1 steps
    for i in range(n_steps - 1) :
        
        print 'Step ' + str(i+1) + ' of ' + str(n_steps)
        
        # loop through all the matrices in the transported_source_dataset.
        for j in range(n_matrices) :
            
            # find the midpoint of the geodesic joining the jth transported
            # source matrix and the i+1th (next) point of disc_geo
            midpoint = make_geodesic(disc_geo[i+1, :, :], transported_source_dataset[j, :, :], d=0.5)
            
            # find the new transported source matrix by moving twice the
            # distance from the ith point of disc geo to the midpoint
            transported_source_dataset[j, :, :] = make_geodesic(midpoint, disc_geo[i, :, :] , d=2.0)
            
        print transported_source_dataset[0, :5, :5]
      
    # final step to the target dataset
    print 'Step ' + str(n_steps) + ' of ' + str(n_steps)
    for j in range(n_matrices) :
            
        # find the midpoint of the geodesic joining the jth transported
        # source matrix and the target mean
        midpoint = make_geodesic(target_mean, transported_source_dataset[j, :, :], d=0.5)
            
        # find the new transported source matrix by moving twice the
        # distance from the ith point of disc geo to the midpoint
        transported_source_dataset[j, :, :] = make_geodesic(midpoint, disc_geo[n_steps - 1, :, :] , d=2.0)     
        
    print transported_source_dataset[0, :5, :5]
        
    # final output: transported source dataset after n_steps steps
    return transported_source_dataset
    
# geodesic distance between two matrices A and B on S++, according to
# information geometry
# From [2]
# AKA affine invariant distance in e.g. [5]
def IG_distance(A, B) :

    A_neg = la.fractional_matrix_power(A, -0.5)
    log_prod = la.logm(A_neg.dot(B).dot(A_neg))
    return np.linalg.norm(log_prod, ord='fro')
    
# geodesic distance between two matrices A and B on S++, according to log-
# Euclidean  metric
# From [6]
def log_Euclidean_distance(A, B) :

    diff_log = la.logm(A) - la.logm(B)
    return np.linalg.norm(diff_log, ord='fro')
        