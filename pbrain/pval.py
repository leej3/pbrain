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



def pval(model_dir,input_csv,output_csv,reference_csv=None,reference_table=None,table_path_out='reference_table.npy',output_dir=None):
    # Load test scores
    test_df = pd.read_csv(input_csv)
    if not 'score' in test_df.columns:
        predict(model_dir,input_csv,output_csv,output_dir)
        test_df = pd.read_csv(output_csv)
    test_scores = test_df['score'].values

    assert reference_csv or reference_table
    if reference_table:
        rtable = np.load(reference_table)
    elif reference_csv:
        # read csv, convert to kde
        # Load training scores
        train_df = pd.read_csv(reference_csv)
        scores = train_df['score'].values # scores for training data
        # Choose bandwidth for Kernel Density Estimation
        iqr = sp.stats.iqr(scores) # use IQR to set maximum bandwidth for KDE
        params = {'bandwidth': np.logspace(-2, 0, 20) * iqr}
        grid = GridSearchCV(KernelDensity(), params, cv = 20)
        grid.fit(scores.reshape((-1,1)))
        bd = grid.best_estimator_.bandwidth
        # Generate reference table
        a,b = np.percentile(scores, [0.1, 99.9])
        xs = np.linspace(a, b, 100)
        # Compute lower tails on reference using KDE with Gaussian kernel
        diffs = np.dot(xs.reshape((-1,1)), np.ones((1, len(scores)))) - np.dot(np.ones((len(xs), 1)), scores.reshape((1,-1))) # a n_test x n_train matrix of differences between xs and training points
        pvals = np.mean(1-norm.cdf(diffs, loc = 0, scale = bd), 1) # the p-values obtained by averaging upper tails of all training points for each grid point
        rtable = np.array([diffs, np.log(pvals)]).T
        np.save(table_path_out, rtable)
    
    #compute pvals from reference table
    pvals = np.exp(sp.interp(test_scores, rtable[:,0], rtable[:, 1]))
    return pvals
