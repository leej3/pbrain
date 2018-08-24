# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import nibabel as nib
import tensorflow as tf
import numpy as np
import os
import time
from nobrainer.volume import zscore
from random import shuffle
import sys

# this code uses the model of neural network to print out 
# the prediction image
def main():
    batch_size = 1  # Number of samples in each batch
    epoch_num = 3    # Number of epochs to train the network
    lr = 0.0001        # Learning rate
    csv = sys.argv[1]
    model_dir = sys.argv[2]
    latent_size = int(sys.argv[3]) # determine the latent size
    output_dir = sys.argv[4]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    # This method returns a variational layer. 

    # input:
    #   parameters: a list of neural network. [0] is the mean layer and [1] is the log of standard
    #   deviation layer. 

    # output:
    #   return: it returns the variational layer of neural network. 
    def sampler(parameters):
        mean = parameters[0]
        log_stddev = parameters[1]
        # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
        std_norm = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean), mean=0, stddev=1)
        
        return mean + tf.keras.backend.exp(log_stddev) * std_norm
      
    # This method returns a convolutional variational neural network.

    # pass latent_vector as input to decoder layers
    # input:
    #   inputs: input layer
    # output:
    #   net: final layer
    #   mean: mean of the variational layer
    #   log_stddev: log_stdev of the variational layer
    def autoencoder(inputs):
        # encoder
        # filter size x dimension x dimension x dimension
        # 1 x 256 x 256 x 256 -> 32 x 256 x 256 x 256 elu
        # 32 x 256 x 256 x 256 -> 16 x 128 x 128 x 128 elu
        # 16 x 128 x 128 x 128 -> 10 x 128 x 128 x 128 elu
        # 10 x 128 x 128 x 128 -> 10 x 64 x 64 x 64 elu
        # 10 x 64 x 64 x 64 -> 10 x 64 x 64 x 64 elu                                                                               
        net = lays.conv3d(inputs, 32, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 16, [3, 3, 3], stride=2, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 10, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 10, [3, 3, 3], stride=2, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 10, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)

        # variational layer
        # mean: 10 x 64 x 64 x 64 -> latent_size x 32 x 32 x 32 elu
        # variance: 10 x 64 x 64 x 64 -> latent_size x 32 x 32 x 32 elu
        # mean + variance -> latent_size x 32 x 32 x 32 elu
        mean = lays.conv3d(net, latent_size, [3, 3, 3], stride=2, padding='SAME', activation_fn=tf.nn.elu)
        log_stddev = lays.conv3d(net, latent_size, [3, 3, 3], stride=2, padding='SAME', activation_fn=tf.nn.elu)
        net = tf.keras.layers.Lambda(sampler)([mean, log_stddev] )

        # decoder
        # latent_size x 32 x 32 x 32 -> laten_size x 64 x 64 x 64 resize
        # latent_size x 64 x 64 x 64 -> 10 x 64 x 64 x 64 elu
        # 10 x 64 x 64 x 64 -> 10 x 128 x 128 x 128 resize
        # 10 x 128 x 128 x 128 -> 10 x 128 x 128 x 128 elu
        # 10 x 128 x 128 x 128 -> 16 x 128 x 128 x 128 elu
        # 16 x 128 x 128 x 128 -> 16 x 256 x 256 x 256 resize
        # 16 x 256 x 256 x 256 -> 32 x 256 x 256 x 256 elu
        # 32 x 256 x 256 x 256 -> 1 x 256 x 256 x 256 linear                                                                           
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = lays.conv3d(net, 10, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = lays.conv3d(net, 10, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 16, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = lays.conv3d(net, 32, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 1, [3, 3, 3], stride=1, padding='SAME')

        return net, mean, log_stddev

    with open(csv, 'r') as csvfile:
        csvlines = csvfile.readlines()[1::]
        csvlines = [item.replace("\n","").split(",") for item in csvlines]

    contents = [item[0] for item in csvlines]
    aseg = [item[1] for item in csvlines]


    ae_inputs = tf.placeholder(tf.float32, (None, 256, 256, 256, 1))  # input to the network (MNIST images)
    ae_outputs, mean, log_stddev = autoencoder(ae_inputs)  # create the Autoencoder network

    # square loss
    recon_loss = tf.keras.backend.sum(tf.keras.backend.square(ae_outputs-ae_inputs))  
    # kl loss
    kl_loss =-0.5 * tf.keras.backend.sum(1 + log_stddev - tf.keras.backend.square(mean) - tf.keras.backend.square(tf.keras.backend.exp(log_stddev)))
    # total loss
    loss = kl_loss + recon_loss

    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initialize the network
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(init)
        # open model
        saver.restore(sess, model_dir + "/model.ckpt")
        # run the sample
        print(contents)
        for i in range(len(contents)):
            orig_address = contents[i]
            aseg_address = aseg[i]

            img = nib.load(orig_address)
            # take zscore 
            orig_img = zscore(np.asarray( img.dataobj ))
            aseg_img = np.asarray( nib.load(aseg_address).dataobj )

            gr = np.greater(aseg_img, 0)
            # take out non-brain parts of the original image
            orig_img = gr * orig_img
            # change shape from 256 x 256 x 256 to (1, 256, 256, 256, 1)
            batch_img =  orig_img[...,None]
            batch_img = np.asarray( [batch_img])

            recon_img, output_loss = sess.run([ae_outputs,loss], feed_dict={ae_inputs: batch_img})

            # assert np.array_equal(recon_img.shape, output_loss.shape)

            print_img = recon_img.reshape( (256,256,256)  )
            # savethe output image ! change the save addres for your use.
            nibImg = nib.spatialimages.SpatialImage(
                dataobj=print_img, affine=img.affine, header=img.header, extra=img.extra)
            file_name = orig_address.split("/")[-1]
            nib.save( nibImg, output_dir+ "/" +file_name.replace(".nii.gz", "_vae_recon.nii.gz") )

if __name__ == '__main__':
    main()
