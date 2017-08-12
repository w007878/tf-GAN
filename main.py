import numpy as np
import tensorflow as tf
import load_data as ldata
import model
import os

from model import BATCH_SIZE
EPOCH_SIZE = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def init_random(shape):
    return np.random.normal(0.0, 20.0, shape)
    
def next_batch(x, y, batch_size=BATCH_SIZE):
    i = 0
    while(i < len(x)):
        yield x[i:i + batch_size], y[i:i + batch_size]
        i = i + batch_size

if __name__ == '__main__':

    sess = tf.Session()
    gan = model.GAN()
    
    images, labels = ldata.load_SVHN()
#    print np.max(images), np.min(images)
#    images = images / 255.

#    ldata.cv2_save(n=10, m=10, data=images[0:100], file_path="meow.png")

    images = (images - 0.5) * 2.
    ldata.cv2_save(n=10, m=10, data=(images[0:100] + 1) / 2., file_path="meow.png")
     
    label_ = tf.placeholder(tf.float32, [None, 1])
    dis_loss = tf.losses.mean_squared_error(labels=label_, predictions=gan.dis)    
    gen_loss = tf.losses.mean_squared_error(labels=label_, predictions=gan.dis_gen)    
    # dis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=gan.dis))
    # gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=gan.dis_gen))

    dis_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gan/dis')
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gan/gen')

    print dis_vars, gen_vars
    
    dis_train_step = tf.train.MomentumOptimizer(0.0002, 0.5).minimize(dis_loss, var_list=dis_vars)
    gen_train_step = tf.train.MomentumOptimizer(0.0002, 0.5).minimize(gen_loss, var_list=gen_vars)
    # dis_train_step = tf.train.MomentumOptimizer(0.0002, 0.5).minimize(dis_loss, var_list=dis_var)
    # gen_train_step = tf.train.MomentumOptimizer(0.0002, 0.5).minimize(gen_loss, var_list=gen_var)

    correct_prediction = tf.equal(tf.sign(label_), tf.sign(gan.dis))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#    print dis_train_step
    
#    print gen_train_step
    sess.run(tf.global_variables_initializer())

    print tf.GraphKeys.TRAINABLE_VARIABLES
    print gen_train_step
    
    for step in range(EPOCH_SIZE):

        batch_step = 0
        for x, _ in next_batch(images,labels):                                             
            batch_step = batch_step + 1

            if len(x) < BATCH_SIZE: break
            input_noise = init_random((BATCH_SIZE, 100))
            xn = gan.gen.eval(session=sess, feed_dict={gan.raw_input_noise:input_noise})
            yn = np.array([[-1]] * BATCH_SIZE)
            
            x_ = np.reshape(x, (BATCH_SIZE, 32 * 32 * 3))
            y = np.array([[1]] * BATCH_SIZE)
            
            # print xn

            rindex = [i for i in range(2 * BATCH_SIZE)]
            np.random.shuffle(rindex)
            tx = np.concatenate((xn, x_))[rindex]
            ty = np.concatenate((yn, y))[rindex]

            train_accuracy = accuracy.eval(session=sess, feed_dict={gan.raw_input_image:tx[0:BATCH_SIZE], label_:ty[0:BATCH_SIZE]})
            
            print("step training accuracy %g" % (train_accuracy))
            #ldata.cv2_save(n=10, m=10, data=(tx[0:100] + 1) / 2., file_path="meow.png")
            
            if batch_step % 100 == 0:
                ldata.cv2_save(n=16, m=16, data=(tx + 1.) / 2., file_path="gen/{}-{}.png".format(step, batch_step))
            
            sess.run(dis_train_step, feed_dict={gan.raw_input_image:tx[0:BATCH_SIZE], label_:ty[0:BATCH_SIZE]})
            # sess.run(dis_train_step, feed_dict={gan.raw_input_image:xn, label_:yn})

            # if batch_step % 20 == 0:
            #     print("Epoch %d, Batch: %d" % (step, batch_step))
            #     input_noise = init_random((BATCH_SIZE, 100))
            #     y = np.array([[1]] * BATCH_SIZE)
            #     sess.run(gen_train_step, feed_dict={gan.raw_input_noise:input_noise, label_:y})

        data = gan.gen.eval(session=sess, feed_dict={gan.raw_input_noise:init_random((BATCH_SIZE, 100))})[0:100]
        
        ldata.cv2_save(n=10, m=10, data=(data + 1.) / 2., file_path="gen/{}.png".format(step))
