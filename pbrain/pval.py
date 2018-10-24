# -*- coding: utf-8 -*-
import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd
from pbrain.models.vae3d import autoencoder
from pbrain.predict import predict
from pbrain.util import get_loss,csv_to_batches, get_image
from pathlib import Path

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
import scipy as sp

# from pbrain.util import zscore



def pval(input_csv,output_csv,reference_csv=None):

    # Load test scores
    test_df = pd.read_csv(input_csv)
    if not 'score' in test_df.columns:
        predict(model_dir,input_csv,output_csv,output_dir)
        test_df = pd.read_csv(output_csv)



    print("Computing pvals...")

    # reconstruction-based pvals
    test_scores = test_df['recon-score'].values
    # Load training scores
    train_df = pd.read_csv(reference_csv)
    scores = train_df['recon-score'].values # scores for training data
    # Choose bandwidth for Kernel Density Estimation
    iqr = sp.stats.iqr(scores) # use IQR to set maximum bandwidth for KDE
    bd = np.std(scores) * len(scores)**(-0.2)
    # Compute lower tails on test data using KDE with Gaussian kernel
    diffs = np.dot(test_scores.reshape((-1,1)), np.ones((1, len(scores)))) - np.dot(np.ones((len(test_scores), 1)), scores.reshape((1,-1))) # a n_test x n_train matrix of differences between test and training points
    pvals = 1 - np.mean(norm.cdf(diffs, loc = 0, scale = bd), 1) # the p-values obtained by averaging lower tails of all training points for each test point
    pvals = 2 * np.minimum(pvals, 1-pvals) # two-sided pvalue
    pvals_recon = pvals


    # KL-based pvals
    test_scores = test_df['kl-score'].values
    # Load training scores
    train_df = pd.read_csv(reference_csv)
    scores = train_df['kl-score'].values # scores for training data
    # Choose bandwidth for Kernel Density Estimation
    iqr = sp.stats.iqr(scores) # use IQR to set maximum bandwidth for KDE
    bd = np.std(scores) * len(scores)**(-0.2)
    # Compute lower tails on test data using KDE with Gaussian kernel
    diffs = np.dot(test_scores.reshape((-1,1)), np.ones((1, len(scores)))) - np.dot(np.ones((len(test_scores), 1)), scores.reshape((1,-1))) # a n_test x n_train matrix of differences between test and training points
    pvals = 1 - np.mean(norm.cdf(diffs, loc = 0, scale = bd), 1) # the p-values obtained by averaging lower tails of all training points for each test point
    pvals = 2 * np.minimum(pvals, 1-pvals) # two-sided pvalue
    pvals_kl = pvals



    pvals = 2 * np.minimum(pvals_recon, pvals_kl)
    test_df['pval'] = pvals
    print(f"Writing csv... {output_csv}")
    test_df.to_csv(output_csv,index = False)


    #compute pvals from reference table
    return pvals
