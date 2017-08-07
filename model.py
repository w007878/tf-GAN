import os
import numpy as np
import tensorflow as tf
import cv2

def init_bias_variable(shape):
    initial = tf.constant(-0.1, shape=shape)
    return tf.Variable(initial)

def init_weight_variable(shape):
    initial = tf.random_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class Discriminator:
    def __init__(self, gen, keep_rate=1.0):
        self.symbol = 0
        self.gen = gen
        self.raw_input_image = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
        self.input_image = tf.reshape(self.raw_input_image + self.gen.h_fc6 * self.symbol, [-1, 32, 32, 3])

        self.W_conv1 = init_weight_variable([3, 3, 3, 32])
        self.b_conv1 = init_bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.input_image, self.W_conv1) + self.b_conv1) 
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        
        self.W_conv2 = init_weight_variable([3, 3, 32, 64])
        self.b_conv2 = init_bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) 

        self.W_conv3 = init_weight_variable([3, 3, 64, 64])
        self.b_conv3 = init_bias_variable([64])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3) + self.b_conv3) 
        self.h_pool3 = max_pool_2x2(self.h_conv3)
        
        self.W_conv4 = init_weight_variable([3, 3, 64, 64])
        self.b_conv4 = init_bias_variable([64])
        self.h_conv4 = tf.nn.relu(conv2d(self.h_pool3, self.W_conv4) + self.b_conv4) 

        self.W_conv5 = init_weight_variable([3, 3, 64, 128])
        self.b_conv5 = init_bias_variable([128])
        self.h_conv5 = tf.nn.relu(conv2d(self.h_conv4, self.W_conv5) + self.b_conv5) 
        self.h_conv5_flat = tf.reshape(self.h_conv5, [-1, 8 * 8 * 128])

        self.W_fc6 = init_weight_variable([8 * 8 * 128, 512])
        self.b_fc6 = init_bias_variable([512])
        self.h_fc6 = tf.nn.relu(tf.matmul(self.h_conv5_flat, self.W_fc6) + self.b_fc6)
        self.h_fc6_drop = tf.nn.dropout(self.h_fc6, keep_rate)
        
        self.W_fc7 = init_weight_variable([512, 1024])
        self.b_fc7 = init_bias_variable([1024])
        self.h_fc7 = tf.nn.relu(tf.matmul(self.h_fc6_drop, self.W_fc7) + self.b_fc7)
        self.h_fc7_drop = tf.nn.dropout(self.h_fc7, keep_rate)
        
        self.W_fc8 = init_weight_variable([1024, 2])
        self.b_fc8 = init_bias_variable([2])
        self.h_fc8 = tf.nn.relu(tf.matmul(self.h_fc7_drop, self.W_fc8) + self.b_fc8)
    
    def set_trainable(self, able):
        self.W_conv1.trainable, self.b_conv1.trainable, self.W_conv2.trainable, self.b_conv2.trainable, \
        self.W_conv3.trainable, self.b_conv3.trainable, self.W_conv4.trainable, self.b_conv4.trainable, \
        self.W_conv5.trainable, self.b_conv5.trainable, self.W_fc6.trainable, self.b_fc6.trainable, \
        self.W_fc7.trainable, self.b_fc7.trainable, self.W_fc8.trainable, self.b_fc8.trainable = [able] * 16
    
    def loss(self, label, logit):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))

class Generator:
    def __init__(self, keep_rate=1.0):
        self.raw_input_image = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
        self.input_image = tf.reshape(self.raw_input_image, [-1, 32, 32, 3])

        self.W_conv1 = init_weight_variable([3, 3, 3, 32])
        self.b_conv1 = init_bias_variable([32])
        self.h_conv1 = tf.nn.relu(conv2d(self.input_image, self.W_conv1) + self.b_conv1) 
        self.h_pool1 = max_pool_2x2(self.h_conv1)
        
        self.W_conv2 = init_weight_variable([3, 3, 32, 64])
        self.b_conv2 = init_bias_variable([64])
        self.h_conv2 = tf.nn.relu(conv2d(self.h_pool1, self.W_conv2) + self.b_conv2) 

        self.W_conv3 = init_weight_variable([3, 3, 64, 64])
        self.b_conv3 = init_bias_variable([64])
        self.h_conv3 = tf.nn.relu(conv2d(self.h_conv2, self.W_conv3) + self.b_conv3) 
        self.h_pool3 = max_pool_2x2(self.h_conv3)
        
        self.W_conv4 = init_weight_variable([3, 3, 64, 128])
        self.b_conv4 = init_bias_variable([128])
        self.h_conv4 = tf.nn.relu(conv2d(self.h_pool3, self.W_conv4) + self.b_conv4) 
        self.h_conv4_flat = tf.reshape(self.h_conv4, [-1, 8 * 8 * 128])

        self.W_fc5 = init_weight_variable([8 * 8 * 128, 256])
        self.b_fc5 = init_bias_variable([256])
        self.h_fc5 = tf.nn.relu(tf.matmul(self.h_conv4_flat, self.W_fc5) + self.b_fc5)
        self.h_fc5_drop = tf.nn.dropout(self.h_fc5, keep_rate)
        
        self.W_fc6 = init_weight_variable([256, 32 * 32 * 3])
        self.b_fc6 = init_bias_variable([32 * 32 * 3])
        self.h_fc6 = tf.nn.relu(tf.matmul(self.h_fc5_drop, self.W_fc6) + self.b_fc6)
        
        # print(np.max(self.h_fc6))
        # self.h_fc6 = self.h_fc6 / np.max(self.h_fc6)

    def generate(self, sess, input_noise):
        # input_noise = init_weight_variable([batch_size, 32 * 32 * 3])
        # print input_noise
        # print self.raw_input_image
        image = sess.run(self.h_fc6, feed_dict={self.raw_input_image:input_noise})
        # image = self.h_fc6.eval(feed_dict={self.raw_input_image:input_noise}, session=sess)
        # image = tf.run(sess, self.h_fc6, feed_dict={self.raw_input_image:input_noise})
        # print self.h_fc
        # print image
        return image
    
    def set_trainable(able):
        self.W_conv1.trainable, self.b_conv1.trainable, self.W_conv2.trainable, self.b_conv2.trainable, \
        self.W_conv3.trainable, self.b_conv3.trainable, self.W_conv4.trainable, self.b_conv4.trainable, \
        self.W_fc5.trainable, self.b_fc5.trainable, self.W_fc6.trainable, self.b_fc6.trainable = [able] * 12
        
class GAN:
    def __init__(self):
        self.gen = Generator()
        self.gan = Discriminator(self.gen)
