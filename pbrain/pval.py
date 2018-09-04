# -*- coding: utf-8 -*-
import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd
from pbrain.models.vae3d import autoencoder
from pbrain.predict import predict
from pbrain.util import get_loss,csv_to_batches, get_image
from pathlib import Path
# from pbrain.util import zscore



def pval(model_dir,input_csv,output_csv,reference_csv,output_dir):
    
    df = pd.read_csv(input_csv)
    if not 'score' in df.columns:
    	predict(model_dir,input_csv,output_csv,output_dir)
    	df = pd.read_csv(output_csv)
    
    	# TODO: df is a dataframe with image path and scores. build empirical
    	# distribution using reference csv and give a pval for input csv