import numpy as np
import tensorflow as tf

def init_weight_variable(name, shape):
    initial = tf.get_variable(name=name, shape=shape, 
              initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
    return initial

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolution_layer(input_data, shape, output_dim, pooling=False):
    W = init_weight_variable(name="weight", shape=shape)
    b = init_weight_variable(name="bais", shape=[output_dim])
    h = tf.nn.relu(conv2d(input_data, W) + b)
    if pooling:
        h_pool = max_pool_2x2(h)
        return h_pool
    else:
        return h

def fc_layer(input_data, input_dim, output_dim, drop_out=False, keep_rate=1.0):
    W = init_weight_variable(name="weight", shape=[input_dim, output_dim])    
    b = init_weight_variable(name="bais", shape=[output_dim])    
    h = tf.nn.relu(tf.matmul(input_data, W) + b)
    if drop_out:
        h_drop =  tf.nn.dropout(h, keep_rate)
        return h_drop
    else:
        return h

def deconvolution_layer(input, filter_size, strides=[1, 1, 1, 1], act=tf.nn.relu6):
    W = init_weight_variable(name="filter", shape=filter_size)
    x_shape = input.shape
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W.shape[2]])
    b = init_weight_variable(name="bais", shape=[x_shape[0], x_shape[1], x_shape[2], W.shape[2]])
    return act(tf.nn.conv2d_transpose(
                value=input, filter=W, output_shape=out_shape, strides=strides
            ) + b)


def Generator(input_noise, BATCH_SIZE=100, keep_rate=1.0):
    with tf.variable_scope("gen"):
        input_image = tf.reshape(input_noise, [BATCH_SIZE, 32, 32, 3])

    with tf.variable_scope("dconv1"):
        h_dconv1 = deconvolution_layer(input_image, filter_size=[3, 3, 16, 3])

    with tf.variable_scope("dconv2"):
        h_dconv2 = deconvolution_layer(h_dconv1, filter_size=[3, 3, 32, 16])

    with tf.variable_scope("dconv3"):
        h_dconv3 = deconvolution_layer(h_dconv2, filter_size=[3, 3, 32, 32])

    with tf.variable_scope("dconv4"):
        h_dconv4 = deconvolution_layer(h_dconv3, filter_size=[3, 3, 32, 32])

    with tf.variable_scope("dconv5"):
        h_dconv5 = deconvolution_layer(h_dconv4, filter_size=[3, 3, 64, 32])

    with tf.variable_scope("dconv6"):
        h_dconv6 = deconvolution_layer(h_dconv5, filter_size=[3, 3, 128, 64])

    with tf.variable_scope("dconv7"):
        h_dconv7 = deconvolution_layer(h_dconv6, filter_size=[3, 3, 256, 128])

    with tf.variable_scope("dconv8"):
        h_dconv8 = deconvolution_layer(h_dconv7, filter_size=[3, 3, 128, 256])
    
    with tf.variable_scope("dconv9"):
        h_dconv9 = deconvolution_layer(h_dconv8, filter_size=[3, 3, 32, 128])
        
    with tf.variable_scope("dconv10"):
        h_dconv10 = deconvolution_layer(h_dconv9, filter_size=[3, 3, 16, 32])
            
    with tf.variable_scope("dconv11"):
        h_dconv11 = deconvolution_layer(h_dconv10, filter_size=[3, 3, 8, 16])
                    
    with tf.variable_scope("dconv12"):
        h_dconv12 = deconvolution_layer(h_dconv11, filter_size=[3, 3, 3, 8])
                    
    print h_dconv11.shape
    print h_dconv12.shape
    return tf.reshape(h_dconv12, [BATCH_SIZE, 32 * 32 * 3])
                    
def Discriminator(raw_input_image, BATCH_SIZE=100, keep_rate=1.0):
    with tf.variable_scope("dis"):
        input_image = tf.reshape(raw_input_image, [BATCH_SIZE, 32, 32, 3])

        with tf.variable_scope("conv1"):
            h_conv1 = convolution_layer(input_image, [3, 3, 3, 32], 32, pooling=True)

        with tf.variable_scope("conv2"):
            h_conv2 = convolution_layer(h_conv1, [3, 3, 32, 64], 64, pooling=False)

        with tf.variable_scope("conv3"):
            h_conv3 = convolution_layer(h_conv2, [3, 3, 64, 64], 64, pooling=True)

        with tf.variable_scope("conv4"):
            h_conv4 = convolution_layer(h_conv3, [3, 3, 64, 64], 64, pooling=False)

        with tf.variable_scope("conv5"):
            h_conv5 = convolution_layer(h_conv4, [3, 3, 64, 128], 128, pooling=False)
            h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 128])
        
        with tf.variable_scope("fc6"):
            h_fc6_drop = fc_layer(h_conv5_flat, 8 * 8 * 128, 512, drop_out=True)
        
        with tf.variable_scope("fc7"):
            h_fc7_drop = fc_layer(h_fc6_drop, 512, 1024, drop_out=True)
            
        with tf.variable_scope("fc8"):
            h_fc8 = fc_layer(h_fc7_drop, 1024, 2)

        return h_fc8
class GAN:
    def __init__(self, BATCH_SIZE=100):
        with tf.variable_scope("gan") as scope:
            # with tf.variable_scope("gan"):
            self.raw_input_image = tf.placeholder(tf.float32, [BATCH_SIZE, 32 * 32 * 3])
            self.raw_input_noise = tf.placeholder(tf.float32, [BATCH_SIZE, 32 * 32 * 3])
            self.gen = Generator(self.raw_input_noise)
            self.dis_gen = Discriminator(self.gen)
            scope.reuse_variables()
            self.dis = Discriminator(self.raw_input_image)
    
