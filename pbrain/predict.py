# -*- coding: utf-8 -*-
import nibabel as nib
import tensorflow as tf
import numpy as np
import pandas as pd
from pbrain.models.vae3d import autoencoder
from pbrain.util import get_loss,csv_to_batches, get_image
from pathlib import Path
# from pbrain.util import zscore


def predict(model_dir,input_csv,output_csv,output_dir):
    contents, batch_per_ep, df = csv_to_batches(input_csv, batch_size=1)

    # lr = 0.0001        # Learning rate

    ae_inputs = tf.placeholder(tf.float32, (None, 256, 256, 256, 1))  # input to the network (MNIST images)
    ae_outputs, mean, log_stddev = autoencoder(ae_inputs)  # create the Autoencoder network

    loss = get_loss(ae_inputs,ae_outputs,mean,log_stddev)

    # predict_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initialize the network
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)
        saver.restore(sess, model_dir + "/model.ckpt")

        for ii,orig_path in enumerate(contents):  # batches loop
            z_img, orig_img = get_image(orig_path)

            # change shape from 256 x 256 x 256 to (1, 256, 256, 256, 1)
            batch_img =  z_img[...,None]
            batch_img = np.asarray( [batch_img])
            
            recon_img, output_loss = sess.run([ae_outputs,loss], feed_dict={ae_inputs: batch_img})
            df.loc[df[df.columns[0]] == orig_path,'score'] = repr(output_loss)
            if output_dir:
                print_img = recon_img.reshape( (256,256,256))
                nibImg = nib.spatialimages.SpatialImage(
                dataobj=print_img, affine=orig_img.affine, header=orig_img.header, extra=orig_img.extra)
                file_name = orig_path.split("/")[-1]
                nib.save( nibImg, output_dir+ "/" +file_name.replace(".nii.gz", "_vae_recon.nii.gz") )
            print(str(ii),'/', str(len(contents)), ': ', orig_path)
            df.to_csv(output_csv,index=False)



