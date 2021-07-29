import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
np.set_printoptions(threshold=np.nan)
RUTA_SALIDA = './modelo_entrenado_mnist_cnn3/'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

import tensorflow as tf

# define a neural network (softmax logistic regression)
x = tf.placeholder(tf.float32, [None, 784], name="X")
x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])


wc1 = tf.Variable(tf.truncated_normal([7,7,1,3], stddev=0.1), name="W_c1")
bc1 = tf.Variable(tf.truncated_normal([3], stddev=0.1), name="b_c1")

wn1 = tf.Variable(tf.truncated_normal([7*7*3, 10], stddev=0.1), name="W_n1")
bn1 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name="b_n1")

# Convolution Layer 7x7 S2
conv1 = tf.nn.conv2d(x_reshape, wc1, strides=[1, 2, 2, 1], padding='SAME')
conv1 = tf.identity(conv1, name="conv1")
conv1b = tf.nn.bias_add(conv1, bc1)
conv1b = tf.identity(conv1b, name="conv1_con_bias")

conv1r = tf.nn.relu(conv1b)
conv1r = tf.identity(conv1r, name="relu")

# AvgPool
pool = tf.nn.avg_pool(conv1r, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
pool = tf.identity(pool, name="pool")

pool1D = tf.reshape(pool, shape=[-1, 7*7*3])
pool1D = tf.identity(pool1D, name="pool1D")
nn1 = tf.matmul(pool1D, wn1)

nn1 = tf.identity(nn1, name="nn1")
logits = nn1 + bn1
logits = tf.identity(logits, name="logits")
y = tf.nn.softmax(logits) # the equation

# define the train step to minimize the cross entropy with SGD
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)

# initialize variables and session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Define metrics
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# train the model mini batch with 100 elements, for 1K times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# evaluate the accuracy of the model in the end
print("Final Test Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# Una vez entrenado el modelo lo guardamos en el disco
inputs = {"X": x}
outputs = {"wc1": wc1,
           "bc1": bc1,
           "wn1": wn1,
           "bn1": bn1,
           "logits": logits}

tf.saved_model.simple_save(sess, RUTA_SALIDA, inputs, outputs)

# Guardamos la salida antes de cerrar la session actual
output_red = sess.run(logits, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_red_session_entrenamiento.txt", 'w') as file:
    file.write(str(output_red))

# Guardamos la salida antes de cerrar la session actual
conv1_np, x_reshape_np, wc1_np = sess.run([conv1, x_reshape, wc1], feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})



with open(RUTA_SALIDA+"output_conv1.txt", 'w') as file:
    file.write(str(conv1_np[0,:,:]))

with open(RUTA_SALIDA+"x_reshape.txt", 'w') as file:
    file.write(str(x_reshape_np))

# Guardamos la salida antes de cerrar la session actual
conv1b_np = sess.run(conv1b, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_conv1b.txt", 'w') as file:
    file.write(str(conv1b_np))

# Guardamos la salida antes de cerrar la session actual
nn1_np = sess.run(nn1, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_nn1.txt", 'w') as file:
    file.write(str(nn1_np))

# Guardamos la salida antes de cerrar la session actual
logits_np = sess.run(logits, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_logits.txt", 'w') as file:
    file.write(str(logits_np))

# Guardamos la salida antes de cerrar la session actual
conv1r_np = sess.run(conv1r, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_conv1r.txt", 'w') as file:
    file.write(str(conv1r_np))

# Guardamos la salida antes de cerrar la session actual
pool_np = sess.run(pool, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_pool.txt", 'w') as file:
    file.write(str(pool_np))

# Guardamos la salida antes de cerrar la session actual
pool1D_np = sess.run(pool1D, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_pool1D.txt", 'w') as file:
    file.write(str(pool1D_np))
