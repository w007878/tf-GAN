import numpy as np
import tensorflow as tf
import load_data as ldata
import model_v4 as model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

    sess = tf.Session()
    gan = model.GAN()
    
    images, labels = ldata.load_SVHN()
    label_ = tf.placeholder(tf.float32, [None, 2])
    
    dis_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=gan.dis))
    gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_, logits=gan.dis_gen))

    dis_train_step = tf.train.AdamOptimizer(1e-2).minimize(dis_loss)
    gen_train_step = tf.train.AdamOptimizer(1e-2).minimize(gen_loss)
    
    sess.run(tf.global_variables_initializer())

    for step in range(EPOCH_SIZE):

        batch_step = 0
        for x, _ in next_batch(images, labels):
            batch_step = batch_step + 1
            input_noise = init_random([BATCH_SIZE, 32 * 32 * 3])
            
            if len(x) < BATCH_SIZE: break
            x_ = x.reshape(BATCH_SIZE * 2, 32 * 32 * 3)[0:BATCH_SIZE]
            y = np.array([[1, 0]] * BATCH_SIZE)

            xn = gan.gen.eval(session=sess, feed_dict={gan.gen.input_noise:input_noise})
            yn = np.array([[0, 1]] * BATCH_SIZE)

            data = np.concatenate((x_, xn))
            label = np.concatenate((y, yn))

            sess.run(dis_train_step, feed_dict={gan.raw_input_image:data, label_:label})

            if batch_step % 20 == 0:
                print("Epoch %d, Batch: %d" % (step, batch_step))
                input_noise = init_random([2 * BATCH_SIZE, 32 * 32 * 3])
                y = np.array([[1, 0]] * 2 * BATCH_SIZE)

                sess.run(gen_train_step, feed_dict={gan.gen.input_noise:input_noise, label_:y})

        if step % 5 == 0:
            data = gan.gen.eval(session=sess, feed_dict={gan.gen.raw_input_image:init_random([100, 32 * 32 * 3])})
            ldata.cv2_save(n=10, m=10, data=data, file_path="gen/{}.png".format(step))
