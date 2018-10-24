# -*- coding: utf-8 -*-
import tensorflow.contrib.layers as lays
import nibabel as nib
import tensorflow as tf
import numpy as np
import time
import pandas as pd
from pathlib import Path
from pbrain.models.vae3d import autoencoder
from pbrain.util import get_batch,get_loss,csv_to_batches,zscore
from random import shuffle
# from pbrain.util import zscore



def train(model_dir,input_csv,batch_size,n_epochs,multi_gpu):
    lr = 0.0001
    contents, batch_per_ep, _ = csv_to_batches(input_csv, batch_size)

    ae_inputs = tf.placeholder(tf.float32, (None, 256, 256, 256, 1))  # input to the network (MNIST images)
    ae_outputs, mean, log_stddev = autoencoder(ae_inputs)  # create the Autoencoder network

    loss, recon_loss, kl_loss = get_loss(ae_inputs,ae_outputs,mean,log_stddev)

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initialize the network
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)
        for ep in range(n_epochs):  # epochs loop
            time2 = time.time()
            shuffle(contents)
            batches = [contents[i:i + batch_size] for i in range(0, len(contents), batch_size) if batch_size == len(contents[i:i + batch_size])]
            i = 0
            for batch in batches:  # batches loop
                time1 = time.time()
                batch_img = get_batch(batch)

                # save model for every 10 samples. 
                if not i%10:
                    save_path = saver.save(sess, model_dir + "/model.ckpt")

                i = i + batch_size
                _, c = sess.run([train_op, recon_loss], feed_dict={ae_inputs: batch_img})

                print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
                print('1 batch took ' + str(time.time() - time1) + ' seconds')
            print('1 epoch took ' + str(time.time() - time2) + ' seconds')
        #file.close()
        # save model 
        save_path = saver.save(sess, model_dir + "/model.ckpt")

