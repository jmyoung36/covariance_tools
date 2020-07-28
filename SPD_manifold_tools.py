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

Notation tries to follow the following conventions:
M, M1, M2...        A Euclidean matrix, indexed Euclidean matrices
S, S1, S2...        An SPD matrix, indexed SPD matrices
MM                  A stack of Euclidean matrices. 1st dimension is n of matrices
SS                  A stack of SPD matrices. 1st dimension is n of matrices


"""

import numpy as np
import scipy.linalg as la

# Generate the tangent vector at M2 'pointing toward' M1, where M1 and M2 are both
# symmetric positive definite matrices in S++. This is the inverse operation of the Exp map.
# From [1]
def log_map(S1, S2) :
    
    S2_pos = la.fractional_matrix_power(S2, 0.5)
    S2_neg = la.fractional_matrix_power(S2, -0.5)
    #return np.linalg.multi_dot([B_pos, la.logm(np.linalg.multi_dot([B_neg, A, B_neg])), B_pos])
    log_prod = la.logm(S2_neg.dot(S1).dot(S2_neg))
    return S2_pos.dot(log_prod).dot(S2_pos)
    
# Project the tangent vector at S2 'pointing toward' S1 back into S++, where S1 
# and S2 are both symmetric positive definite matrices. This is the inverse operation of the Log map.
# From [1]
def exp_map(S1, S2) :
    
    S2_pos = la.fractional_matrix_power(S2, 0.5)
    S2_neg = la.fractional_matrix_power(S2, -0.5)
    #return np.linalg.multi_dot([B_pos, la.expm(np.linalg.multi_dot([B_neg, A, B_neg])), B_pos])
    exp_prod = la.expm(S2_neg.dot(S1).dot(S2_neg))
    return S2_pos.dot(exp_prod).dot(S2_pos)

# create a geodesic, i.e. local shortest path, from matrix S2 to matrix S1 where 
# S1 and S2 are both positive definite matrices on S++. Take an optional distance argument 
# d allowing us to travel e.g. half way from S2 to S1, or twice the distance from
# S2 to S1.
# From [1]
def make_geodesic(S1, S2, d=1):
    
    return exp_map(d * log_map(S1, S2), S2)

# calculate the geometric mean of a set of covariance matrices and use this to project
# the matrices into the tangent space of the mean matrix
def project_to_mean_tangent(SS) :
    
    # find geometric mean
    # construct base covariance matrix by repeated averaging in tangent space
    # first, initialise base covariance matrix
    M_base = np.mean(SS, axis=0)
    
    for i in range(2) :
        
        #print base_cov_matrix[:5, :5]
    
        # project all matrices into the tangent space
        MM_tangent = np.zeros_like(SS)
        for j in range(np.shape(SS)[2]) :
        
            MM_tangent[j, :, :] = np.linalg.multi_dot([la.fractional_matrix_power(M_base, 0.5), la.logm(np.linalg.multi_dot([la.fractional_matrix_power(M_base, -0.5), SS[:, :, j],la.fractional_matrix_power(M_base, -0.5)])), la.fractional_matrix_power(M_base, 0.5)])
    
        # calculate the tangent space mean
        M_tangent_mean = np.mean(MM_tangent, axis=0)
        
        #print tangent_space_base_cov_matrix[:5, :5]
        
        # project new tangent mean back to the manifold
        M_base = np.linalg.multi_dot([la.fractional_matrix_power(M_base, 0.5), la.expm(np.linalg.multi_dot([la.fractional_matrix_power(M_base, -0.5), M_tangent_mean ,la.fractional_matrix_power(M_base, -0.5)])), la.fractional_matrix_power(M_base, 0.5)])
        
    # apply whitening transport and projection for training AND testing data
#    projected_matrices = np.zeros_like(matrices)   
#    base_cov_matrix_pow = la.fractional_matrix_power(base_cov_matrix, -0.5)
#    
#    for i in range(len(matrices)) :
#        projected_matrices[i, :, :] = la.logm(np.linalg.multi_dot([base_cov_matrix_pow, matrices[i, :, :], base_cov_matrix_pow]))
#       
#    return projected_matrices
    return M_base

# calculate the Log-Euclidean mean of a set of n dxd matrices in S++, by applying
# a matrix logarithm to project the matrices onto the tangent space at I,
# calculating the arithmetic mean, and then projecting it back into S++.
# Take a stack of n dxd matrices MM, as a numpy array with dimensions d x d x n
# From [1]
def log_Euclidean_mean(SS):
    
    # create a structure to store the tangent space matrices
    MM = np.zeros_like(SS)
    
    # project all the matrices in MM into the tangent space
    for i in range(np.shape(SS)[2]) :
        
        MM[i, :, :] = np.logm(SS[i, :, :])
        
    # take the mean and project back to S++
    MM_mean = np.mean(MM, axis=0)
    return la.expm(MM_mean)

# calculate the geometric mean of a set of n dxd matrices in S++. In the 
# general case, initially calculate the Euclidean mean. Then project the
# matrices into the tangent space of the Euclidean mean with log_map, calculate
# the mean of the projected matrices, and project this back into S++ with 
# exp_map to provide an updated estimate of the geometric mean. Repeat with 
# this new estimate until the estimate converges or the maximum number of 
# iterations is reached. Optionally weight the mean of the projected matrices.
# Take a stack of n dxd matrices SS, as a numpy array with dimensions n x d x d
# From [3, 4] for general case and 2 for matrix case respectively 
def geometric_mean(SS, tol=10e-10, max_iter=50, weights=None):
    
    # number of matrices
    n = np.shape(SS)[0]
    if n == 2 and isinstance(weights, type(None)) : 
        
        # special case - just take midpoint of geodesic joining the matrices [5]
        S1 = np.matrix(np.squeeze(SS[0, :, :]))
        S2 = np.matrix(np.squeeze(SS[1, :, :]))
        S_mean = S1 ** 0.5 * ((S1 ** -0.5 * S2 ** S1 **-0.5) ** 0.5) * S1 **0.5
        return S_mean
    
    else:
        
        # general case, more than 2 matrices
       
        # initialise variables
        # Euclidean mean as first estimate
        if isinstance(weights, type(None)) :
        
            new_est = np.mean(SS, axis=0)
            
        else :
            
            # use broadcasting for weighted sum
            new_est = np.sum(weights[:, np.newaxis, np.newaxis] * SS, axis = 0)
            
        # convergence criterion
        crit = np.finfo(np.float64).max
        k = 0

    
        # start the loop
        while (crit > tol) and (k < max_iter):
        
        
            #print new_est[:5, :5]
        
            # update the current estimate
            current_est = new_est
        
            # project all the matrices into the tangent space of the current estimate
            MM_tangent = np.zeros_like(SS)
            for i in range(n) :
            
                MM_tangent[i, :, :] = log_map(SS[i, :, :], current_est)
            
            # arithmetic (optionally weighted) mean in the tangent space
            if isinstance(weights, type(None)) :
                
                M_tangent_mean = np.mean(MM_tangent, axis=0)
                
            else :
            
                # use broadcasting for weighted sum
                M_tangent_mean = np.sum(weights[:, np.newaxis, np.newaxis] * MM_tangent, axis = 0)
        
            #print S[:5, :5]
        
            # project S back to S++ to create a new estimated mean
            new_est = exp_map(M_tangent_mean, current_est)
            #new_est = exp_map(current_est, S)
        
            # housekeeping: update k and crit
            k = k + 1
            #crit = np.linalg.norm(S, ord='fro')
            crit = np.linalg.norm(new_est - current_est, ord='fro')
            print (k)
            print (crit)
        
        return new_est

# adaptation of local synthethic instances (LSI) method (Brown C.J. et al. 
# (2015) Prediction of Motor Function in Very Preterm Infants Using Connectome 
# Features and Local Synthetic Instances)
# adapted to generate synthetic SPD matrices
# generate weights and use the geometric_mean method to weight samples in the 
# tangent space
# interpolate target variables too in Euclidean space using standard LSI.
def manifold_LSI(SS, Y, params) :
    
    assert isinstance(SS, np.ndarray), 'Input SS must be a Numpy array'
    assert isinstance(Y, np.ndarray), 'Input Y must by a Numpy array'
    
    n = SS.shape[0]

    # get parameters
    max_num_weighted_samples = params.get('max_num_weighted_samples', np.inf)
    num_synthetic_instances = params.get('num_synthetic_instances', n)
    num_weighted_samples = min(n, max_num_weighted_samples)
    
    if 'p_range' in params:
        p_range = params['p_range']
    else:
        p_range = (1.2, 3.0)
        
    assert len(p_range) == 2, 'p_range must have exactly two elements: (p_min, p_max)'
    p_min = p_range[0]
    p_max = p_range[1]

    b = np.array(range(num_weighted_samples)) + 1.0

    # allocate memory for synthetic isntances
    synth_SS = np.zeros([num_synthetic_instances] + list(SS.shape[1:]))
    synth_Y = np.zeros([num_synthetic_instances] + list(Y.shape[1:]))
    
    for i in range(num_synthetic_instances):
        
        # generarate weights
        t = float(i) / num_synthetic_instances
        p_val = p_min + t * (p_max - p_min)
        w = 1.0 / np.power(b, p_val)
        w /= sum(w)

        # generate random sample of data & labels
        inds = np.random.permutation(n)[:num_weighted_samples]
        SS_sample = SS[inds, :, :]
        Y_sample = Y[inds, :]
        
        # make synthetic matrix using weighted geometric_mean
        synth_SS[i, :, :] = geometric_mean(SS_sample, max_iter=100, weights = w)
        
        # make synthetic labels using weighted mean
        synth_Y [i, :] = np.sum(w[:, np.newaxis] * Y_sample, axis=0)

    return synth_SS, synth_Y

# take a source and a target mean and an SPD connectivity matrix. Generate a 
# discretised geodesic from the source mean to the target mean. Use this 
# geodesic to parallel transport the connectivity matrices in the 
# source dataset to the location of the target dataset with Schild's Ladder.
# algorith takem from [1]
def transport_schilds_ladder(S_source, S_target, SS, n_steps=10) :
    
    # generate a discretised geodesic with n_steps step, at source_mean and 
    # pointing toward target_mean
    n, d, d = np.shape(SS)
    disc_geo = np.zeros((n_steps, d, d))
    for i in range(n_steps) :
        
        frac_dist = (i+1)/float(n_steps)
        disc_geo[i, :, :] = make_geodesic(S_target, S_source, frac_dist)
        
    # initialise the transported matrices
    # copy original matrices BY VALUE 
    SS_transported = np.copy(SS)
        
    # perform the transport
    print ("Performing parallel transport with Schild's ladder...")
    # first n_steps - 1 steps
    for i in range(n_steps - 1) :
        
        print ('Step ' + str(i+1) + ' of ' + str(n_steps))
        
        # loop through all the matrices in the transported_source_dataset.
        for j in range(n) :
            
            # find the midpoint of the geodesic joining the jth transported
            # source matrix and the i+1th (next) point of disc_geo
            S_midpoint = make_geodesic(disc_geo[i+1, :, :], SS_transported[j, :, :], d=0.5)
            
            # find the new transported source matrix by moving twice the
            # distance from the ith point of disc geo to the midpoint
            SS_transported[j, :, :] = make_geodesic(S_midpoint, disc_geo[i, :, :] , d=2.0)
            
        print (SS_transported[0, :5, :5])
      
    # final step to the target dataset
    print ('Step ' + str(n_steps) + ' of ' + str(n_steps))
    for j in range(n) :
            
        # find the midpoint of the geodesic joining the jth transported
        # source matrix and the target mean
        S_midpoint = make_geodesic(S_target, SS_transported[j, :, :], d=0.5)
            
        # find the new transported source matrix by moving twice the
        # distance from the ith point of disc geo to the midpoint
        SS_transported[j, :, :] = make_geodesic(S_midpoint, disc_geo[n_steps-1, :, :] , d=2.0)     
        
    print (SS_transported[0, :5, :5])
        
    # final output: transported source dataset after n_steps steps
    return SS_transported
    
# take a source and a target mean and a stack of SPD connectivity matrices. 
# Transport the matrices from the source to the target mean with the closed
# from parallel transport described in [4]
def transport_yair(S_source, S_target, SS) :
    
    # initialise memory for transported SS
    SS_transported = np.zeros_like(SS)
    
    # calculate intermediate matrix E
    E = la.fractional_matrix_power(np.dot(S_target, la.inv(S_source)), 0.5)
    
    # loop through the matrices to transport each in turn
    # eq 7 in the reference paper
    for i in range(np.shape(SS)[0]) :
        
        S = SS[i, :, :]
        S_transported = np.dot(np.dot(E, S), np.transpose(E))
        SS_transported[i, :, :] = S_transported
        
    return SS_transported


# geodesic distance between two matrices S1 and S2 on S++, according to
# information geometry
# From [2]
# AKA affine invariant distance in e.g. [5]
def IG_distance(S1, S2) :

    S1_neg = la.fractional_matrix_power(S1, -0.5)
    log_prod = la.logm(S1_neg.dot(S2).dot(S1_neg))
    return np.linalg.norm(log_prod, ord='fro')
    
# geodesic distance between two matrices A and B on S++, according to log-
# Euclidean  metric
# From [6]
def log_Euclidean_distance(S1, S2) :

    diff_log = la.logm(S1) - la.logm(S2)
    return np.linalg.norm(diff_log, ord='fro')
        