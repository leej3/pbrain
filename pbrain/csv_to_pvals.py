# -*- coding: utf-8 -*-
import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd
from pbrain.models.vae3d import autoencoder
from pbrain.predict import predict
from pbrain.pval import pval
from pbrain.train import train
from pbrain.util import get_loss,csv_to_batches, get_image, clean_csv, check_path_is_writable
import pbrain
from pathlib import Path

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
import scipy as sp

# from pbrain.util import zscore

def csv_to_pvals(input_csv,model_dir=None,output_dir=None,output_csv=None,reference_csv=None,clean_input_csv=True):
    # Set defaults
    if not output_csv:
        output_csv = input_csv
    if not reference_csv:
        prbain_dir = Path(pbrain.__file__).parent
        reference_csv = prbain_dir / 'reference_files' / 'reference.csv'
    if not model_dir:
        model_dir = Path(pbrain.__file__).parent / 'reference_files' /'model'

    # Check that pval column does not already exist in csv
    if Path(output_csv).exists():
        out_df = pd.read_csv(output_csv)
        if 'pval' in out_df.columns:
            raise(ValueError, "output_csv already contains a pval column.")

    # Clean csv if required
    check_path_is_writable(output_csv)
    if clean_input_csv:
        clean_csv(input_csv, output_csv)
        pred_in_csv = output_csv
    else:
        pred_in_csv = input_csv
    # Predict if required
    test_df = pd.read_csv(pred_in_csv)
    if not 'score' in test_df.columns:
        predict(model_dir,input_csv,output_csv,output_dir)
        pval_in_csv = output_csv
    else:
        pval_in_csv = pred_in_csv

    # Compute pvalues and write to output csv
    pval(pval_in_csv,output_csv,reference_csv=reference_csv)

    
