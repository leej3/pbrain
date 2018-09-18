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



def pval(model_dir,input_csv,output_csv,reference_csv=None,output_dir=None):
    # Load test scores
    test_df = pd.read_csv(input_csv)
    if not 'score' in test_df.columns:
        predict(model_dir,input_csv,output_csv,output_dir)
        test_df = pd.read_csv(output_csv)
    test_scores = test_df['score'].values

    train_df = pd.read_csv(reference_csv)
    scores = train_df['score'].values # scores for training data
    # Choose bandwidth for Kernel Density Estimation
    iqr = sp.stats.iqr(scores) # use IQR to set maximum bandwidth for KDE
    params = {'bandwidth': np.array([0.1, 0.2, 0.3, 0.4]) * iqr}
    grid = GridSearchCV(KernelDensity(), params, cv = 4)
    grid.fit(scores.reshape((-1,1)))
    bd = grid.best_estimator_.bandwidth
    # Generate reference table
    a,b = np.percentile(scores, [0.1, 99.9])
    xs = test_scores
    diffs = np.dot(xs.reshape((-1,1)), np.ones((1, len(scores)))) - np.dot(np.ones((len(xs), 1)), scores.reshape((1,-1))) # a n_test x n_train matrix of differences between xs and training points
    pvals = np.mean(1-norm.cdf(diffs, loc = 0, scale = bd), 1) # the p-values obtained by averaging upper tails of all training points for each grid point
    #return pvals
    df['pvalue'] = pvals
    df.to_csv(output_csv)
