
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

def main():
    batch_size = 1  # Number of samples in each batch
    epoch_num = 20    # Number of epochs to train the network # it usaually takes a day to train for 1 epoch on GTX Titan XP
    lr = 0.0001        # Learning rate
    csv = sys.argv[1]
    model_dir = sys.argv[2]
    latent_size = int(sys.argv[3]) # determine the latent size

    # This method returns a batch and its labels. 

    # input: 
    #   content: list of string address of original images.
    #   aseg: list of string address of segmentation images.
    #   i : location to extract the batches from. 

    # output: 
    #   imgs: 5D numpy array of image btaches. Specifically, it takes the shape of (batch_size, 256, 256, 256, 1)
    #   arrName: list of string addresses that are labels for imgs. 
    def get_batch(content, aseg, i):
        arr = []
        arrName = []
        for j in range(i, i + batch_size):


            orig_img = zscore(np.asarray( nib.load(content[j]).dataobj ))
            aseg_img = np.asarray( nib.load(aseg[j]).dataobj )

            # take out non-brain parts of the original image.
            gr = np.greater(aseg_img, 0)
            orig_img = gr * orig_img
            
            arr.append( orig_img)
        imgs = np.array(arr)
        imgs = imgs[..., None]
        return imgs, arrName

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
        net = lays.conv3d(inputs, 32, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu) #  32
        net = lays.conv3d(net, 16, [3, 3, 3], stride=2, padding='SAME', activation_fn=tf.nn.elu) #  16
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
        # 16 x 128} x 128 x 128 -> 16 x 256 x 256 x 256 resize
        # 16 x 256 x 256 x 256 -> 32 x 256 x 256 x 256 elu
        # 32 x 256 x 256 x 256 -> 1 x 256 x 256 x 256 linear
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = lays.conv3d(net, 10, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = lays.conv3d(net, 10, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = lays.conv3d(net, 16, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu) # 16
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = lays.conv3d(net, 32, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu) # 32
        net = lays.conv3d(net, 1, [3, 3, 3], stride=1, padding='SAME')

        return net, mean, log_stddev

    with open(csv, 'r') as f:
        csvlines = f.readlines()[1::]
        csvlines = [item.replace("\n","").split(",") for item in csvlines]

    contents = [item[0] for item in csvlines]
    aseg = [item[1] for item in csvlines]

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
        for ep in range(epoch_num):  # epochs loop
            time2 = time.time()
            i = 0
            for batch_n in range(batch_per_ep):  # batches loop
                time1 = time.time()
                batch_img, batch_names = get_batch(contents, aseg, i)

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


if __name__ == '__main__':
    main()
