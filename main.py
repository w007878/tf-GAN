import os
import tensorflow as tf
import numpy as np
import cv2
import load_data
import model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
BATCH_SIZE = 50
EPOCH_SIZE = 10000

def next_batch(x, y, batch_size=BATCH_SIZE):
    i = 0
    while(i < len(x)):
        yield x[i:i + batch_size], y[i:i + batch_size]
        i = i + batch_size

if __name__ == '__main__':
    network = model.GAN()
    images, labels = load_data.load_SVHN()
    
    label_ = tf.placeholder(tf.float32, [None, 10])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(network.dis.loss(labels=network.dis.h_fc8, logits=label_))
    correct_prediction = tf.equal(tf.argmax(label_, 1), tf.argmax(sim_network.h_fc8, 1))
    
    for step in range(EPOCH_SIZE):
        network.dis.set_trainable(True)
        for x, _ in next_batch(images, labels):
            y = np.ones(len(x))
            xn = network.gen.generate(sess, len(x))
            yn = np.zeros(len(s))
            data = np.concat(x, xn)
            label = np.concat(y, yn)
            sess.run(train_step, feep_dict={network.dis.raw_input_image:data.reshape([BATCH_SIZE, 32 * 32 * 3]),\
                                            label_:label})
        network.dis.set_trainable(False)
        y = np.ones(BATCH_SIZE)
        sess.run(train_step, feed_dict={network.gen.raw_input_image:model.init_weight_variable([BATCH_SIZE, 32 * 32 * 3]),\
                                        network.dis.raw_input_images:network.gen.h_fc6, label_:y})
        if step % 500 == 0:
            xn = network.gen.generate(sess, 100)
            load_data.cv2_save(n=10, m=10, data, file_path="{}.png" % step)
    