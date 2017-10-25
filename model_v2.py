import tensorflow as tf

import numpy as np
import os
from tensorflow.contrib.layers import conv2d, dropout, max_pool2d, flatten, fully_connected, avg_pool2d, conv2d_transpose
from tensorflow.python.training.training import AdadeltaOptimizer, AdamOptimizer, FtrlOptimizer, MomentumOptimizer

BATCH_SIZE = 128

def default_conv2d(inputs, output_dim, name='conv2d'):
    return conv2d(inputs, output_dim, kernel_size=[5, 5], stride=[2, 2],
        padding='SAME', activation_fn=None, 
        biases_initializer=tf.constant_initializer(0.0),
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        scope=name
    )

def default_fc(inputs, output_dim, name='fc'):
    return fully_connected(inputs, output_dim, 
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        biases_initializer=tf.constant_initializer(0.0),
        scope=name
    )

def default_conv2d_transpose(inputs, num_outputs, name='conv2d_transpose'):
    return conv2d_transpose(inputs, num_outputs, 
        kernel_size=[5, 5], stride=[2, 2], 
        activation_fn=None,
        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
        biases_initializer=tf.constant_initializer(0.0),
        scope=name
    )

# The generator model    
def Generator(input_data):
    # input is a [BATCH_SIZE, 2048] random tensor / np.array
    
    with tf.variable_scope("gen"):
        
        # First layer: Fully connection
        fc_1 = default_fc(input_data, 1024 * 4 * 4, 'fc')
        fc_1 = tf.reshape(fc_1, [BATCH_SIZE, 4, 4, 1024])

        # Four Convolution Layers
        deconv1 = tf.nn.relu(default_conv2d_transpose(fc_1, 512, 'deconv1'))
        deconv2 = tf.nn.relu(default_conv2d_transpose(deconv1, 256, 'deconv2'))
        deconv3 = tf.nn.relu(default_conv2d_transpose(deconv2, 128, 'deconv3'))
        deconv4 = conv2d_transpose(deconv3, 3, kernel_size=[5, 5], stride=[1, 1], 
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        biases_initializer=tf.constant_initializer(0.0),
                        scope='deconv4'
                  )
                
        
        res = flatten(deconv4)
    return res

# The discriminator model
def Discriminator(input_data):
    with tf.variable_scope("dis"):
        image = tf.reshape(input_data, [BATCH_SIZE, 32, 32, 3])

        conv1 = tf.nn.leaky_relu(default_conv2d(image, 32, 'conv1'))
        conv2 = tf.nn.leaky_relu(default_conv2d(conv1, 64, 'conv2'))
        conv3 = tf.nn.leaky_relu(default_conv2d(conv2, 128, 'conv3'))
        conv4 = tf.nn.leaky_relu(default_conv2d(conv3, 256, 'conv4'))
        conv4_f = flatten(conv4)
        
        fc1 = tf.nn.leaky_relu(default_fc(conv4_f, 128, name='fc_1'))
        fc2 = default_fc(fc1, 2, name='fc_2')

    return fc2

class GAN:
    def __init__(self):
        with tf.variable_scope("gan") as scope:
            # with tf.variable_scope("gan"):
            self.raw_input_image = tf.placeholder(tf.float32, [BATCH_SIZE, 32 * 32 * 3])
            self.raw_input_noise = tf.placeholder(tf.float32, [BATCH_SIZE, 2048])
            self.gen = Generator(self.raw_input_noise)
            self.dis_gen = Discriminator(self.gen)
            scope.reuse_variables()
            self.dis = Discriminator(self.raw_input_image)
            