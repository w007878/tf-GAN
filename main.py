import os
import tensorflow as tf
import numpy as np
import cv2
import load_data
import model 

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
BATCH_SIZE = 50
EPOCH_SIZE = 10000

def init_random(shape):
    return np.random.random_sample(shape)
    
def next_batch(x, y, batch_size=BATCH_SIZE):
    i = 0
    while(i < len(x)):
        yield x[i:i + batch_size], y[i:i + batch_size]
        i = i + batch_size

if __name__ == '__main__':

    network = model.GAN()
    images, labels = load_data.load_SVHN()
    
    label_ = tf.placeholder(tf.float32, [None, 2])
    # sess = tf.Session()
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(network.dis.loss(logit=network.dis.h_fc8, label=label_))
    correct_prediction = tf.equal(tf.argmax(label_, 1), tf.argmax(network.dis.h_fc8, 1))
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    for step in range(EPOCH_SIZE):
        if step % 10 == 0:
            print("Epoch %d" % step)
            
        network.dis.set_trainable(True)
        batch_step = 0
        for x, _ in next_batch(images, labels):

            if batch_step % 100 == 0:
                print("Batch: %d" % batch_step)
            batch_step = batch_step + 1
            
            x_ = x.reshape(len(x), 32 * 32 * 3)
            y = np.array([[1, 0]] * len(x))

            input_noise = init_random([len(x), 32 * 32 * 3])
            xn = network.gen.generate(sess, input_noise=input_noise)
            yn = np.array([[0, 1]] * len(x))

            x_ = x_.astype(np.float32)
            x_ = x_ / np.max(x_)
            xn = xn / np.max(xn)
            
            # print xn.shape, x_.shape
            # print yn.shape, y.shape
            data = np.concatenate((x_, xn))
            label = np.concatenate((y, yn))
            
            # print data.shape, label.shape
            sess.run(train_step, feed_dict={network.dis.raw_input_image:data, label_:label})

        input_noise = init_random([len(x), 32 * 32 * 3])
        network.dis.set_trainable(False)
        y = np.array([[1, 0]] * BATCH_SIZE)
        sess.run(train_step, feed_dict={network.gen.raw_input_image:input_noise,\
                                        network.dis.raw_input_image:(network.gen.h_fc6 / np.max(network.gen.h_fc6)),\
                                        label_:y})
        if step % 500 == 0:
            data = network.gen.generate(sess, init_random([100, 32 * 32 * 3]))
            load_data.cv2_save(n=10, m=10, data=data, file_path="gen/{}.png".format(step))
    