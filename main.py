import os
import tensorflow as tf
import numpy as np
import cv2
import load_data
import model 

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
BATCH_SIZE = 50
EPOCH_SIZE = 1000

def init_random(shape):
    return np.random.random_sample(shape)
    
def next_batch(x, y, batch_size=BATCH_SIZE):
    i = 0
    while(i < len(x)):
        yield x[i:i + batch_size], y[i:i + batch_size]
        i = i + batch_size

if __name__ == '__main__':

    tmp_buff = open('tmp.out', 'a')
    
    gan = model.GAN()
    images, labels = load_data.load_SVHN()
    
    label_ = tf.placeholder(tf.float32, [None, 2])
    # sess = tf.Session()
    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(gan.gan.loss(logit=gan.gan.h_fc8, label=label_))
    correct_prediction = tf.equal(tf.argmax(label_, 1), tf.argmax(gan.gan.h_fc8, 1))
    
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    
    for step in range(EPOCH_SIZE):
        if step % 10 == 0:
            print("Epoch %d" % step)
            
        gan.gan.set_trainable(True)
        batch_step = 0
        for x, _ in next_batch(images, labels):
                
            batch_step = batch_step + 1
            
            x_ = x.reshape(len(x), 32 * 32 * 3)
            y = np.array([[1, 0]] * len(x))

            input_noise = init_random([len(x), 32 * 32 * 3])
            xn = gan.gen.generate(sess, input_noise=input_noise)
            yn = np.array([[0, 1]] * len(x))

            x_ = x_.astype(np.float32)
            x_ = x_ / np.max(x_)
            xn = xn / np.max(xn)
            
            # print xn.shape, x_.shape
            # print yn.shape, y.shape
            data = np.concatenate((x_, xn))
            label = np.concatenate((y, yn))
            gan.gen.set_trainable(False)
            
            # print data.shape, label.shape
            gan.symbol = 0
            sess.run(train_step, feed_dict={gan.gan.raw_input_image:data, label_:label, \
                                            gan.gen.raw_input_image:np.zeros([2 * len(x), 32 * 32 * 3]),})

            if batch_step % 100 == 0:
                gan.symbol = 1
                print("Epoch %d, Batch: %d" % (step, batch_step))
                input_noise = init_random([BATCH_SIZE, 32 * 32 * 3])
                gan.gen.set_trainable(True)
                gan.gan.set_trainable(False)
                y = np.array([[1, 0]] * BATCH_SIZE)
                
                sess.run(train_step, feed_dict={gan.gen.raw_input_image:input_noise, \
                                                gan.gan.raw_input_image:np.zeros([BATCH_SIZE, 32 * 32 * 3]), \
                                                label_:y})
                gan.gan.set_trainable(True)
                # gan.transform(True)
                gan.symbol = 0
        
        tmp_buff.write(gan.gen.W_conv1)
        tmp_buff.write('\n')
        tmp_buff.write(gan.gen.b_conv1)
        tmp_buff.write('\n')
                
        if step % 20 == 0:
            data = gan.gen.generate(sess, init_random([100, 32 * 32 * 3]))
            load_data.cv2_save(n=10, m=10, data=data, file_path="gen/{}.png".format(step))
    
    tmp_buff.close()