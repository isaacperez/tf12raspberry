import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf


with tf.Session() as sess:

    #x = tf.ones([1,2,2,1], dtype=tf.float32)
    x = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], shape=[1,4,4,1],dtype=tf.float32)
    #w = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49], shape=[7,7,1,1], dtype=tf.float32)
    #conv = tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], padding='SAME')
    #conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    #conv_np, x_np, w_np = sess.run([conv, x, w])
    conv_np, x_np, = sess.run([conv, x])
    print("X:")
    print(x_np)
    #print("W:")
    #print(w_np)
    print("CONV:")
    print(conv_np)
    print(conv_np.shape)
