# -*- coding: utf-8 -*-
import tensorflow.contrib.layers as lays
import nibabel as nib
import tensorflow as tf
import numpy as np
import time
from pbrain.volume import zscore
import pandas as pd
from pathlib import Path
from pbrain.models.vae3d import autoencoder
# from pbrain.util import zscore


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
        orig_img = zscore(np.asarray( nib.load(content[j]).dataobj ))
        arr.append( orig_img)
    imgs = np.array(arr)
    imgs = imgs[..., None]
    return imgs

      
def train(model_dir,csv,batch_size,n_epochs,multi_gpu):
    lr = 0.0001        # Learning rate

    df = pd.read_csv(csv)

    df['exists'] = df[df.columns[0]].apply(lambda x: Path(x).exists())
    df = df.query('exists')
    contents = df[df.columns[0]].sample(frac=1)

    # make the image list a multiple of batch_size
    contents = contents[0: len(contents) // (batch_size) * batch_size]

    # calculate the number of batches per epoch
    batch_per_ep = len(contents) // batch_size

    ae_inputs = tf.placeholder(tf.float32, (None, 256, 256, 256, 1))  # input to the network (MNIST images)
    ae_outputs, mean, log_stddev = autoencoder(ae_inputs)  # create the Autoencoder network

    # square loss
    recon_loss = tf.keras.backend.sum(tf.keras.backend.square(ae_outputs-ae_inputs))  
    # kl loss
    kl_loss =-0.5 * tf.keras.backend.sum(1 + log_stddev - tf.keras.backend.square(mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
    #total loss
    loss = kl_loss + recon_loss

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initialize the network
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    with tf.Session() as sess:

        sess.run(init)
        for ep in range(n_epochs):  # epochs loop
            time2 = time.time()
            i = 0
            for batch_n in range(batch_per_ep):  # batches loop
                time1 = time.time()
                batch_img = get_batch(contents, i, batch_size)

                # save model for every 10 samples. 
                if not i%10:
                    save_path = saver.save(sess, model_dir + "/model.ckpt")

                print(contents[i:i+batch_size])

                i = i + batch_size
                _, c = sess.run([train_op, recon_loss], feed_dict={ae_inputs: batch_img})

                print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))
                print('1 batch took ' + str(time.time() - time1) + ' seconds')
            print('1 epoch took ' + str(time.time() - time2) + ' seconds')
        #file.close()
        # save model 
        save_path = saver.save(sess, model_dir + "/model.ckpt")

