import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 128

# use leakly relu instead of relu as the activative function
def leakly_relu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

# generate a random tensor in the given shape
def random_init(name, shape, mean=0., stddev=0.02):
    initial = tf.get_variable(name=name, shape=shape, 
              initializer=tf.random_normal_initializer(mean, stddev))
    return initial
    
# build a convolution layer
def transpose_conv(x, W, output_shape, strides=[1, 2, 2, 1]):
    return tf.nn.conv2d_transpose(x, filter=W, output_shape=output_shape, strides=strides, padding='SAME')

def conv_layer(x, W, strides=[1, 2, 2, 1]):
    return tf.nn.conv2d(x, filter=W, strides=strides, padding="SAME")

def stack_bias(W, name="bias"):
    return random_init(name, W)

# The generator model    
def Generator(input):
    # input is a [BATCH_SIZE, 100] random tensor / np.array
    
    with tf.variable_scope("gen"):
        
        # First layer: Fully connection
        with tf.variable_scope("fc_layer"):
            fc_W = random_init("weight", [100, 1024 * 4 * 4])
            fc_b = random_init("bias", [1024 * 4 * 4])
            fc_h = leakly_relu(tf.matmul(input, fc_W) + fc_b)
            fc_h = tf.reshape(fc_h, [BATCH_SIZE, 4, 4, 1024])

        # Four Convolution Layers
        with tf.variable_scope("conv1"):
            conv1_W = random_init("filter", [5, 5, 512, 1024])
            conv1_b = stack_bias([BATCH_SIZE, 8, 8, 512])
            conv1_h = leakly_relu(transpose_conv(fc_h, conv1_W, [BATCH_SIZE, 8, 8, 512]) + conv1_b)
        
        with tf.variable_scope("conv2"):
            conv2_W = random_init("filter", [5, 5, 256, 512])
            conv2_b = stack_bias([BATCH_SIZE, 16, 16, 256])
            conv2_h = leakly_relu(transpose_conv(conv1_h, conv2_W, [BATCH_SIZE, 16, 16, 256]) + conv2_b)

        with tf.variable_scope("conv3"):
            conv3_W = random_init("filter", [5, 5, 128, 256])
            conv3_b = stack_bias([BATCH_SIZE, 32, 32, 128])
            conv3_h = leakly_relu(transpose_conv(conv2_h, conv3_W, [BATCH_SIZE, 32, 32, 128]) + conv3_b)

        with tf.variable_scope("conv4"):
            conv4_W = random_init("filter", [5, 5, 3, 128])
            conv4_b = stack_bias([BATCH_SIZE, 32, 32, 3])
            conv4_h = tf.nn.tanh(transpose_conv(conv3_h, conv4_W, [BATCH_SIZE, 32, 32, 3], 
                                                strides=[1, 1, 1, 1]) + conv4_b)

    return tf.reshape(conv4_h, [BATCH_SIZE, 32 * 32 * 3])

# The discriminator model
def Discriminator(input):
    with tf.variable_scope("dis"):
        image = tf.reshape(input, [BATCH_SIZE, 32, 32, 3])

        with tf.variable_scope("conv1"):
            conv1_W = random_init("filter", [5, 5, 3, 32])
            conv1_b = stack_bias([BATCH_SIZE, 16, 16, 32])
            conv1_h = leakly_relu(conv_layer(image, conv1_W) + conv1_b)
        
        with tf.variable_scope("conv2"):
            conv2_W = random_init("filter", [5, 5, 32, 64])
            conv2_b = stack_bias([BATCH_SIZE, 8, 8, 64])
            conv2_h = leakly_relu(conv_layer(conv1_h, conv2_W) + conv2_b)

        with tf.variable_scope("conv3"):
            conv3_W = random_init("filter", [5, 5, 64, 128])
            conv3_b = stack_bias([BATCH_SIZE, 4, 4, 128])
            conv3_h = leakly_relu(conv_layer(conv2_h, conv3_W) + conv3_b)

        with tf.variable_scope("conv4"):
            conv4_W = random_init("filter", [5, 5, 128, 256])
            conv4_b = stack_bias([BATCH_SIZE, 2, 2, 256])
            conv4_h = tf.nn.tanh(conv_layer(conv3_h, conv4_W) + conv4_b)
            conv4_flat = tf.reshape(conv4_h, [BATCH_SIZE, 2 * 2 * 256])
            
        with tf.variable_scope("fc1"):
            fc1_W = random_init("weight", [2 * 2 * 256, 128])
            fc1_b = random_init("bias", [128])
            fc1_h = leakly_relu(tf.matmul(conv4_flat, fc1_W) + fc1_b)
        
        with tf.variable_scope("fc2"):
            fc2_W = random_init("weight", [128, 1])
            fc2_b = random_init("bias", [1])
            fc2_h = tf.nn.tanh(tf.matmul(fc1_h, fc2_W) + fc2_b)

    return fc2_h

class GAN:
    def __init__(self):
        with tf.variable_scope("gan") as scope:
            # with tf.variable_scope("gan"):
            self.raw_input_image = tf.placeholder(tf.float32, [BATCH_SIZE, 32 * 32 * 3])
            self.raw_input_noise = tf.placeholder(tf.float32, [BATCH_SIZE, 100])
            self.gen = Generator(self.raw_input_noise)
            self.dis_gen = Discriminator(self.gen)
            scope.reuse_variables()
            self.dis = Discriminator(self.raw_input_image)
    