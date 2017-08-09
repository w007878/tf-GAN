import os
import tensorflow as tf
import numpy as np
import cv2
import load_data
import model_v3 as model

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
    
    sess = tf.Session()
    
    gan = model.GAN()
    images, labels = load_data.load_SVHN()
    
    label_ = tf.placeholder(tf.float32, [None, 2])
    image_ = tf.placeholder(tf.float32, [None, 32 * 32 * 3])
    # sess = tf.Session()
    
    # dis_train_step = tf.train.AdamOptimizer(1e-4).minimize(gan.dis_loss(sess, input_image=image_, labels=label_))
    # gen_train_step = tf.train.AdamOptimizer(1e-4).minimize(gan.gen_loss(sess, input_noise=image_))
    # correct_prediction = tf.equal(tf.argmax(label_, 1), tf.argmax(gan.h_fc8, 1))
    
    sess.run(tf.global_variables_initializer())
    
    for step in range(EPOCH_SIZE):
        if step % 10 == 0:
            print("Epoch %d" % step)
            
        batch_step = 0
        for x, _ in next_batch(images, labels):
            batch_step = batch_step + 1
            
            if len(x) < BATCH_SIZE: break
            x_ = x.reshape(BATCH_SIZE, 32 * 32 * 3)
            y = np.array([[1, 0]] * BATCH_SIZE)

            # input_noise = init_random([BATCH_SIZE, 32 * 32 * 3])
            xn = gan.gen.generate(sess)[0:BATCH_SIZE]
            yn = np.array([[0, 1]] * BATCH_SIZE)

            x_ = x_.astype(np.float32)
            # x_ = x_ / np.max(x_)
            # xn = xn / np.max(xn)
            
            # print xn.shape, x_.shape
            # print yn.shape, y.shape
            data = np.concatenate((x_, xn))
            label = np.concatenate((y, yn))

            dis_train_step = tf.train.AdamOptimizer(1e-4).minimize(\
                                                    tf.reduce_mean(\
                                                    tf.nn.softmax_cross_entropy_with_logits(\
                                                    labels=label, \
                                                    logits=gan.dis.h_fc8.eval(session=sess, \
                                                    feed_dict={gan.dis.raw_input_image:data}))))
            
            # print data.shape, label.shape
            # gan.symbol = 0
            sess.run(dis_train_step, feed_dict={image_:data, label_:label})

            if batch_step % 100 == 0:
                print("Epoch %d, Batch: %d" % (step, batch_step))
                input_noise = init_random([2 * BATCH_SIZE, 32 * 32 * 3])
                y = np.array([[1, 0]] * 2 * BATCH_SIZE)

                gen_train_step = tf.train.AdamOptimizer(1e-4).minimize(\
                                                        tf.reduce_mean(\
                                                        tf.nn.softmax_cross_entropy_with_logits(\
                                                        labels=y, \
                                                        logits=gan.dis.h_fc8.eval(session=sess, \
                                                        feed_dict={gan.dis.raw_input_image:\
                                                        gan.gen.generate(sess, input_noise)}))))
                
                # gen_train_step = tf.train.AdamOptimizer(1e-4).minimize(\
                # gan.gen_loss(sess, input_noise=image_))
                sess.run(gen_train_step)
                # gan.transform(True)
        
        # print gan.gen.W_conv1, gan.gen.b_conv1, gan.gen.W_fc5, gan.gen.b_fc5, "================"
        # tmp_buff.write(gan.gen.W_conv1)
        # tmp_buff.write('\n')
        # tmp_buff.write(gan.gen.b_conv1)
        # tmp_buff.write('\n')
                
        if step % 5 == 0:
            data = gan.gen.generate(sess, init_random([100, 32 * 32 * 3]))
            load_data.cv2_save(n=10, m=10, data=data, file_path="gen/{}.png".format(step))
    
    tmp_buff.close()