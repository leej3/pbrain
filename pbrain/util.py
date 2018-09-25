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

# sys.excepthook = lambda exctype,exc,traceback : print("{}: {}".format(exctype.__name__,exc))


def clean_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    df['loads'] = df[df.columns[0]].apply(check_nibload)
    (df.query('loads').
        drop('loads', inplace=False).
        to_csv(output_csv, index=False, sep=',')
     )


def conform_csv(input_csv, output_csv, output_shape, voxel_dims):
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

    output_shape : tuple of ints
        X,Y, and Z dimensions of the output image.

    voxel_dims : list of ints
        X,Y, and Z dimensions of the voxels of the output image.
    """
    df = pd.read_csv(input_csv)
    out_scan_paths = []
    scan_col = 0
    for ii, scan_path in enumerate(df.iloc[:, scan_col]):
        img = nib.load(scan_path)
        # Check image requires conformation
        if (img.shape != output_shape) or (img.header.get_zooms() != voxel_dims):
            conformed = conform_scan(img=img,
                                     output_shape=output_shape,
                                     voxel_dims=voxel_dims)
            conformed.header.set_zooms((voxel_dims))
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


def zscore(a):
    """Return array of z-scored values."""
    a = np.asarray(a)
    return (a - a.mean()) / a.std()


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


def get_image(image_path, image_shape=(256, 256, 256), voxel_dims=[1, 1, 1]):
    orig_img = nib.load(image_path)
    if orig_img.shape != image_shape:
        conform_scan(img=orig_img, image_shape=image_shape)
    z_img = zscore(orig_img.get_data())
    return z_img, orig_img


def get_batch(content, i, batch_size):
    # This method returns a batch and its labels.

    # input:
    #   content: list of string address of original images.
    #   i : location to extract the batches from.

    # output:
    #   imgs: 5D numpy array of image btaches. Specifically, it takes the shape of (batch_size, 256, 256, 256, 1)
    #   arrName: list of string addresses that are labels for imgs.
    arr = []
    for j in range(i, i + batch_size):
        z_img, _ = get_image(content[j])
        arr.append(z_img)
    imgs = np.array(arr)
    imgs = imgs[..., None]
    return imgs


def csv_to_batches(csv, batch_size):
    df = pd.read_csv(csv)

    df['exists'] = df[df.columns[0]].apply(lambda x: Path(x).exists())
    df = df.query('exists')
    contents = df[df.columns[0]].sample(frac=1)

    # make the image list a multiple of batch_size
    contents = contents[0: len(contents) // (batch_size) * batch_size]

    # calculate the number of batches per epoch
    batch_per_ep = len(contents) // batch_size
    return contents, batch_per_ep, df


def get_loss(ae_inputs, ae_outputs, mean, log_stddev):
    # square loss
    recon_loss = tf.keras.backend.sum(
        tf.keras.backend.square(ae_outputs - ae_inputs)) / 2.0
    # kl loss
    kl_loss = -0.5 * tf.keras.backend.sum(1 + 2.0 * log_stddev - tf.keras.backend.square(
        mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
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


def conform_image(img, output_shape=(256, 256, 256), voxel_dims=[1, 1, 1]):
    """
    Imitation of mri_convert from freesurfer. Consists of minimal processing
    for pipelines involving neural networks. The default output is an image
    that is resampled to 256 voxels in each dimension with a voxel size of 1mm
    cubed. The last column of the image affine is scaled according to the
    scaling factor of the resampling.

    Parameters
    ----------
    img : nibabel.nifti1.Nifti1Image
        An alternative to providing scan_path. 
    output_shape : tuple of 3 ints
        number of voxels in each dimension.
    voxel_dims : list
        Length in mm for x,y, and z dimensions of each voxel.

    Returns
    -------
    resampled_img : nibabel.nifti1.Nifti1Image
        The resampled image.
    """
    # Resample for a voxel size of voxel_dims
    resampled_nib = nibabel.processing.resample_to_output(
        img, voxel_sizes=voxel_dims)
    # use resample_img to resize to output dimensions defined by user
    target_affine = img.affine.copy()
    target_affine[:3, 3] = target_affine[:3, 3] * output_shape / img.shape
    resampled_img = image.resample_img(
        resampled_nib, target_shape=output_shape, target_affine=target_affine)
    return resampled_img
