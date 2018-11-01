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
from nilearn import image, datasets
from nibabel import processing
#import nibabel.spatialimages.HeaderDataError

# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))


def clean_csv(input_csv, output_csv):
    print('Cleaning csv file of images that cannot be loaded by nibabel')
    df = pd.read_csv(input_csv)
    df['loads'] = df[df.columns[0]].apply(check_nibload)
    (df.query('loads').
        drop('loads', inplace=False,axis = 1).
        to_csv(output_csv, index=False, sep=',')
     )
    not_used = ' '.join(df.query("~loads")[df.columns[0]])
    print(f"Not using {not_used}")


def conform_csv(input_csv, output_csv, target_shape, voxel_dims):
    """
    With an input csv of scans, returns a csv of paths to scans that have been
    conformed as described by mri_convert in freesurfer. Briefly, returns a
    cubic volume of isotropic voxels.
    input_csv : str
        Path to input csv in which the first column consists of input scans
        for the neural network.

    output_csv : str
        Path to output csv in which the first column consists of input scans
        for the neural network that have been conformed to cubic images with
        isotropic voxel dimensions.

    target_shape : tuple of ints
        X,Y, and Z dimensions of the output image.

    voxel_dims : list of ints
        X,Y, and Z dimensions of the voxels of the output image.
    """
    print(f"Resampling scans to voxel_dims of {voxel_dims} and shape of {target_shape}")
    df = pd.read_csv(input_csv)
    out_scan_paths = []
    scan_col = 0
    for ii, scan_path in enumerate(df.iloc[:, scan_col]):
        img = nib.load(scan_path)
        # Check image requires conformation
        if (img.shape != target_shape) or not np.isclose(np.array(img.header.get_zooms()),np.array(voxel_dims)).all():
            try:
                conformed = conform_image(
                    img=img,
                    target_shape=target_shape,
                    voxel_dims=voxel_dims)
            # except nibabel.spatialimages.HeaderDataError:
            except:
                print("Error: Not including {scan_path}")
                df.drop([ii])
                nib.save(conformed, conformed_path)
                continue

            # Write image to disk
            suffix_string = '_conformed' + ''.join(Path(scan_path).suffixes)
            conformed_path = (scan_path.split('.')[0] + suffix_string)
            df.iloc[ii, scan_col] = conformed_path
            nib.save(conformed, conformed_path)
        df.to_csv(output_csv, index=False)


def check_path_is_writable(p, path_type='output csv'):
    if not os.access(p, mode=os.W_OK):
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
        print("Failure, could not read this file: ", input_path)
        return False
    return True

def zscore_with_stats(a,stats_path):
    """Return array of z-scored values."""
    a = np.asarray(a)
    df = pd.read_csv(stats_path + '/input_stats.csv') 
    mean = df['mean'][0]
    std = df['std'][0]
    if std == 0:
        std = 10**-7
    return (a - mean) / std


def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    std = a.std()
    if std == 0:
        std = 10**-7
    return (a - a.mean()) / std


def run_cmd(cmd):
    import subprocess
    pp = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print([v.split('//')[-1] for v in pp.stderr.decode('utf-8').splitlines()])
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


def get_image(image_path, image_shape=(256, 256, 256), voxel_dims=[1, 1, 1],stats_path=None):
    orig_img = nib.load(image_path)
    if orig_img.shape != image_shape:
        conform_image(img=orig_img, target_shape=image_shape)
    if stats_path:
        z_img = zscore_with_stats(orig_img.get_data(),stats_path)
    else:
        z_img = zscore(orig_img.get_data())
    return z_img, orig_img


def get_batch(batch,stats_path=None):
    # This method returns a batch and its labels.

    # input:
    #   batch: list of string address of original images.

    # output:
    #   imgs: 5D numpy array of image btaches. Specifically, it takes the shape of (batch_size, 256, 256, 256, 1)
    
    arr = []
    for path in batch:
        z_img, _ = get_image(path,stats_path=stats_path)
        arr.append(z_img)
    imgs = np.array(arr)
    imgs = imgs[..., None]
    return imgs


def csv_to_batches(csv, batch_size):
    df = pd.read_csv(csv)

    df['exists'] = df[df.columns[0]].apply(lambda x: Path(x).exists())
    df = df.query('exists')
    contents = list(df[df.columns[0]])

    # calculate the number of batches per epoch
    batch_per_ep = len(contents) // batch_size
    return contents, batch_per_ep, df

def get_loss_0(ae_inputs, ae_outputs, mean, log_stddev):
    # square loss
    recon_loss = tf.keras.backend.sum(
        tf.keras.backend.square(ae_outputs - ae_inputs)) / 2.0
    # abs loss
    #recon_loss = tf.keras.backend.sum(
        #tf.keras.backend.abs(ae_outputs - ae_inputs)) / 2.0**(-0.5)
    # kl loss
    kl_loss = -0.5 * tf.keras.backend.sum(1 + log_stddev - tf.keras.backend.square(
        mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
    # total loss
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss

def get_loss(ae_inputs, ae_outputs, mean, log_stddev, stats_path):
    # square loss
    recon_loss = tf.keras.backend.sum(
        tf.keras.backend.square(ae_outputs - ae_inputs)) / 2.0
    # abs loss
    #recon_loss = tf.keras.backend.sum(
        #tf.keras.backend.abs(ae_outputs - ae_inputs)) / 2.0**(-0.5)
    # kl loss
    prior_mean = tf.convert_to_tensor(np.load(stats_path + '/prior_mean.npy'), dtype=tf.float32)
    prior_std = tf.convert_to_tensor(np.load(stats_path + '/prior_std.npy'), dtype=tf.float32)
    kl_loss = tf.keras.backend.sum(-0.5 - log_stddev + (tf.keras.backend.square(
        mean-prior_mean) + tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))/(2.0*tf.square(prior_std)))
    # total loss
    loss = recon_loss + kl_loss
    return loss, recon_loss, kl_loss


def get_loss_old(ae_inputs, ae_outputs, mean, log_stddev):
    # square loss
    recon_loss = tf.keras.backend.sum(
        tf.keras.backend.square(ae_outputs - ae_inputs))
    # kl loss
    kl_loss = -0.5 * tf.keras.backend.sum(1 + log_stddev - tf.keras.backend.square(
        mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
    # total loss
    loss = kl_loss + recon_loss
    return loss


def rescale_affine(input_affine, voxel_dims=[1, 1, 1], target_center_coords= None):
    """
    This function uses a generic approach to rescaling an affine to arbitrary
    voxel dimensions. It allows for affines with off-diagonal elements by
    decomposing the affine matrix into u,s,v (or rather the numpy equivalents)
    and applying the scaling to the scaling matrix (s).

    Parameters
    ----------
    input_affine : np.array of shape 4,4
        Result of nibabel.nifti1.Nifti1Image.affine
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.
    target_center_coords: list of float
        3 numbers to specify the translation part of the affine if not using the same as the input_affine.

    Returns
    -------
    target_affine : 4x4matrix
        The resampled image.
    """
    # Initialize target_affine
    target_affine = input_affine
    # Decompose the image affine to allow scaling
    u,s,v = np.linalg.svd(target_affine[:3,:3],full_matrices=False)
    
    # Rescale the image to the appropriate voxel dimensions
    s = voxel_dims
    
    # Reconstruct the affine
    target_affine[:3,:3] = u @ np.diag(s) @ v

    # Set the translation component of the affine computed from the input
    # image affine if coordinates are specified by the user.
    if target_center_coords is not None:
        target_affine[:3,3] = target_center_coords
    return target_affine


def test_rescale_affine():
    input_img = datasets.load_mni152_template()


def conform_image(img, target_shape=(256, 256, 256), voxel_dims=[1.0, 1.0, 1.0]):
    """
    Imitation of mri_convert from freesurfer. Consists of minimal processing
    for pipelines involving neural networks. The default output is an image
    that is resampled to 256 voxels in each dimension with a voxel size of 1mm
    cubed.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        An alternative to providing scan_path. 
    target_shape : tuple of 3 ints
        number of voxels in each dimension.
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.

    Returns
    -------
    resampled_img : nibabel.nifti1.Nifti1Image
        The resampled image.
    """
    if not img.header['sform_code'] >0:
        raise ValueError("The image header must contain a valid sform affine.")
    # initialize the affine of the output image
    target_affine = img.affine.copy()
    
    # Calculate the translation part of the affine
    spatial_dimensions = (img.header['dim'] * img.header['pixdim'])[1:4]
    
    # Calculate the translation affine as a proportion of the real world
    # spatial dimensions
    image_center_as_prop = img.affine[0:3,3] / spatial_dimensions
    
    # Calculate the equivalent center coordinates in the target image
    dimensions_of_target_image = (np.array(voxel_dims) * np.array(target_shape))
    target_center_coords =  dimensions_of_target_image * image_center_as_prop 

    target_affine = rescale_affine(target_affine,voxel_dims,target_center_coords)
    # Resample the image
    resampled_img = image.resample_img(img, target_affine=target_affine,target_shape=target_shape)
    resampled_img.header.set_zooms((np.absolute(voxel_dims)))
    return resampled_img
