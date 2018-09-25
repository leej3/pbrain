# -*- coding: utf-8 -*-
"""Utilities."""
import numpy as np
import nibabel as nib
import pandas as pd
import sys
import tensorflow as tf
from pathlib import Path
import argparse
import os

# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))
def clean_csv(input_csv,output_csv):
	df = pd.read_csv(input_csv)
	df['loads'] = df[df.columns[0]].apply(check_nibload)
	(df.query('loads').
		drop('loads',inplace=False).
		to_csv(output_csv,index=False,sep=',')
		)


def check_path_is_writable(p,path_type= 'output csv'):
    if not os.access(p,mode = os.W_OK):
        raise ValueError(f"Do not have write access to {p}"
                        "which has been specified as the {path_type} path")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def check_nibload(input_path):

	try:
		nib.load(input_path).get_data()
	except Exception as e:
		print(e)
		print("Failure: ", input_path)
		return False
	return True



def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    return (a - a.mean()) / a.std()


def run_cmd(cmd):
    import subprocess
    pp = subprocess.run(cmd,shell=True,stdout=subprocess.PIPE,stderr= subprocess.PIPE)
    print([v.split('//')[-1] for v in pp.stderr.decode('utf-8').splitlines() ])
    return pp


def setup_exceptionhook():
    """
    Overloads default sys.excepthook with our exceptionhook handler.

    If interactive, our exceptionhook handler will invoke pdb.post_mortem;
    if not interactive, then invokes default handler.
    """
    def _pdb_excepthook(type, value, tb):
        if sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty():
            import traceback
            import pdb
            traceback.print_exception(type, value, tb)
            # print()
            pdb.post_mortem(tb)
        else:
            print(
              "We cannot setup exception hook since not in interactive mode")

    sys.excepthook = _pdb_excepthook


def get_image(image_path):
	orig_img =  nib.load(image_path)
	z_img = zscore(orig_img.get_data())
	return z_img, orig_img

def get_batch(content, i,batch_size):
    # This method returns a batch and its labels. 

    # input: 
    #   content: list of string address of original images.
    #   i : location to extract the batches from. 

    # output: 
    #   imgs: 5D numpy array of image btaches. Specifically, it takes the shape of (batch_size, 256, 256, 256, 1)
    #   arrName: list of string addresses that are labels for imgs. 
    arr = []
    for j in range(i, i + batch_size):
        z_img,_ = get_image(content[j])
        arr.append( z_img)
    imgs = np.array(arr)
    imgs = imgs[..., None]
    return imgs

def csv_to_batches(csv,batch_size):
	df = pd.read_csv(csv)

	df['exists'] = df[df.columns[0]].apply(lambda x: Path(x).exists())
	df = df.query('exists')
	contents = df[df.columns[0]].sample(frac=1)

	# make the image list a multiple of batch_size
	contents = contents[0: len(contents) // (batch_size) * batch_size]

	# calculate the number of batches per epoch
	batch_per_ep = len(contents) // batch_size
	return contents, batch_per_ep, df

def get_loss(ae_inputs,ae_outputs,mean,log_stddev):
    # square loss
    recon_loss = tf.keras.backend.sum(tf.keras.backend.square(ae_outputs-ae_inputs))/2.0  
    # kl loss
    kl_loss =-0.5 * tf.keras.backend.sum(1 + 2.0*log_stddev - tf.keras.backend.square(mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
    #total loss
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss
      
def get_loss_old(ae_inputs,ae_outputs,mean,log_stddev):
    # square loss
    recon_loss = tf.keras.backend.sum(tf.keras.backend.square(ae_outputs-ae_inputs))  
    # kl loss
    kl_loss =-0.5 * tf.keras.backend.sum(1 + log_stddev - tf.keras.backend.square(mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
    #total loss
    loss = kl_loss + recon_loss
    return loss
