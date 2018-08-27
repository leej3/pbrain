import tensorflow.contrib.layers as lays
import tensorflow as tf

def sampler(parameters):
    # This method returns a variational layer. 

    # input:
    #   parameters: a list of neural network. [0] is the mean layer and [1] is the log of standard
    #   deviation layer. 

    # output:
    #   return: it returns the variational layer of neural network. 
    mean = parameters[0]
    log_stddev = parameters[1]
    # we sample from the standard normal a matrix of batch_size * latent_size (taking into account minibatches)
    std_norm = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mean), mean=0, stddev=1)
    
    return mean + tf.keras.backend.exp(log_stddev) * std_norm


def autoencoder(inputs):
    # This method returns a convolutional variational neural network.

    # pass latent_vector as input to decoder layers
    # input:
    #   inputs: input layer
    # output:
    #   net: final layer
    #   mean: mean of the variational layer
    #   log_stddev: log_stdev of the variational layer
    
    # encoder
    # filter size x dimension x dimension x dimension
    # 1 x 256 x 256 x 256 -> 32 x 256 x 256 x 256 elu
    # 32 x 256 x 256 x 256 -> 16 x 128 x 128 x 128 elu
    # 16 x 128 x 128 x 128 -> 10 x 128 x 128 x 128 elu
    # 10 x 128 x 128 x 128 -> 10 x 64 x 64 x 64 elu
    # 10 x 64 x 64 x 64 -> 10 x 64 x 64 x 64 elu
    latent_size = 10
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

