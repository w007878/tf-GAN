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

def Generator(raw_input_image, BATCH_SIZE=100, keep_rate=1.0):
    with tf.variable_scope("gen"):
        input_image = tf.reshape(raw_input_image, [BATCH_SIZE, 32, 32, 3])

        with tf.variable_scope("conv1"):
            h_conv1 = convolution_layer(input_image, [3, 3, 3, 32], 32, pooling=True)

        with tf.variable_scope("conv2"):
            h_conv2 = convolution_layer(h_conv1, [3, 3, 32, 64], 64, pooling=False)

        with tf.variable_scope("conv3"):
            h_conv3 = convolution_layer(h_conv2, [3, 3, 64, 64], 64, pooling=True)

        with tf.variable_scope("conv4"):
            h_conv4 = convolution_layer(h_conv3, [3, 3, 64, 128], 128, pooling=False)
            h_conv4_flat = tf.reshape(h_conv4, [-1, 8 * 8 * 128])            

        with tf.variable_scope("fc5"):
            h_fc5_drop = fc_layer(h_conv4_flat, 8 * 8 * 128, 256, drop_out=True)
        
        with tf.variable_scope("fc6"):
            h_fc6 = fc_layer(h_fc5_drop, 256, 32 * 32 * 3)

        return h_fc6

class GAN:
    def __init__(self):
        with tf.variable_scope("gan") as scope:
            # with tf.variable_scope("gan"):
            self.raw_input_image = tf.placeholder(tf.float32, [100, 32 * 32 * 3])
            self.gen = Generator(self.raw_input_image)
            self.dis_gen = Discriminator(self.gen)
            scope.reuse_variables()
            self.dis = Discriminator(self.raw_input_image)
    
