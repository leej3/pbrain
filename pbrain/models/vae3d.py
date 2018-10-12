import tensorflow.contrib.layers as lays
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    gf_dim = 20
    if True:
        net = tf.nn.elu(instance_norm(conv3d(inputs, gf_dim, 7, 2, padding='SAME', name='g_e1_c'), 'g_e1_bn'))
        net = tf.nn.elu(instance_norm(conv3d(net, gf_dim*2, 3, 2, padding='SAME',name='g_e2_c'), 'g_e2_bn'))
        net = tf.nn.elu(instance_norm(conv3d(net, gf_dim*4, 3, 2,padding='SAME', name='g_e3_c'), 'g_e3_bn'))
        
        net = residule_block(net, gf_dim*4, name='g_r1')
        net = residule_block(net, gf_dim*4, name='g_r2')
        net = residule_block(net, gf_dim*4, name='g_r3')
        net = residule_block(net, gf_dim*4, name='g_r4')
        
        mean = lays.conv3d(net, gf_dim*4, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        log_stddev = lays.conv3d(net, gf_dim*4, [3, 3, 3], stride=1, padding='SAME', activation_fn=tf.nn.elu)
        net = tf.keras.layers.Lambda(sampler)([mean, log_stddev] )
        
        net = residule_block(net, gf_dim*4, name='g_r5')
        net = residule_block(net, gf_dim*4, name='g_r6')
        net = residule_block(net, gf_dim*4, name='g_r7')
        net = residule_block(net, gf_dim*4, name='g_r8')
        
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = conv3d(net, gf_dim*2, 3, 1,padding='SAME', name='g_d1_dc')
        net = tf.nn.elu(instance_norm(net, 'g_d1_bn'))
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = conv3d(net, gf_dim, 3, 1,padding='SAME', name='g_d2_dc')
        net = tf.nn.elu(instance_norm(net, 'g_d2_bn'))
        net = tf.keras.backend.resize_volumes(net, 2, 2, 2, "channels_last")
        net = conv3d(net, 1, 7, 1, padding='SAME', name='g_pred_c')
    
    return net, mean, log_stddev

def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[4]
        scale = tf.get_variable("scale", [depth], initializer=tf.contrib.layers.xavier_initializer())
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv3d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv3d"):
    with tf.variable_scope(name):
        return slim.conv3d(input_, output_dim, ks, s, padding=padding, activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer(),biases_initializer=None)
    
def residule_block(x, dim, ks=3, s=1, name='res'):
    p = int((ks - 1) / 2)
    y = tf.pad(x, [[0, 0], [p, p], [p, p], [p, p], [0, 0]], "REFLECT")
    y = instance_norm(conv3d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
    y = tf.pad(tf.nn.elu(y), [[0, 0], [p, p], [p, p], [p, p], [0, 0]], "REFLECT")
    y = instance_norm(conv3d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
    return y + x