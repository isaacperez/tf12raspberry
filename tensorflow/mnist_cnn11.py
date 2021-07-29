import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import shutil
np.set_printoptions(threshold=np.nan)
RUTA_SALIDA = './modelo_entrenado_mnist_cnn11/'
if os.path.exists(RUTA_SALIDA):
    shutil.rmtree(RUTA_SALIDA)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

import tensorflow as tf

# define a neural network (softmax logistic regression)
x = tf.placeholder(tf.float32, [None, 784], name="X")
x_reshape = tf.reshape(x, shape=[-1, 28, 28, 1])


wc1 = tf.Variable(tf.truncated_normal([1,1,1,4], stddev=0.1), name="W_c1")
bc1 = tf.Variable(tf.truncated_normal([4], stddev=0.1), name="b_c1")

wc2 = tf.Variable(tf.truncated_normal([7,7,4,8], stddev=0.1), name="W_c2")
bc2 = tf.Variable(tf.truncated_normal([8], stddev=0.1), name="b_c2")

wn1 = tf.Variable(tf.truncated_normal([14*14*8, 10], stddev=0.1), name="W_n1")
bn1 = tf.Variable(tf.truncated_normal([10], stddev=0.1), name="b_n1")

# Conv1
conv1 = tf.nn.conv2d(x_reshape, wc1, strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.identity(conv1, name="conv1")
conv1b = tf.nn.bias_add(conv1, bc1)
conv1b = tf.identity(conv1b, name="conv1_con_bias")
conv1r = tf.nn.relu(conv1b)
conv1r = tf.identity(conv1r, name="conv1_con_activacion")

# Conv2
conv2 = tf.nn.conv2d(conv1r, wc2, strides=[1, 2, 2, 1], padding='SAME')
conv2 = tf.identity(conv2, name="conv2")
conv2b = tf.nn.bias_add(conv2, bc2)
conv2b = tf.identity(conv2b, name="conv2_con_bias")
conv2r = tf.nn.relu(conv2b)
conv2r = tf.identity(conv2r, name="conv2_con_activacion")

conv2r_1D = tf.reshape(conv2r, shape=[-1, 14*14*8])
conv2r_1D = tf.identity(conv2r_1D, name="conv2_con_activacion_1D")

# NN
nn1 = tf.matmul(conv2r_1D, wn1)
nn1 = tf.identity(nn1, name="nn1")

logits = nn1 + bn1
logits = tf.identity(logits, name="logits")

y = tf.nn.softmax(logits) # the equation

# define the train step to minimize the cross entropy with SGD
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(0.005).minimize(cross_entropy)

# initialize variables and session
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Define metrics
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# train the model mini batch with 100 elements, for 1K times
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(128)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    if i % 100 == 0:
        print("Test Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# evaluate the accuracy of the model in the end
print("Final Test Accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


# Una vez entrenado el modelo lo guardamos en el disco
inputs = {"X": x}
outputs = {"wc1": wc1,
           "bc1": bc1,
           "wc2": wc2,
           "bc2": bc2,
           "wn1": wn1,
           "bn1": bn1,
           "logits": logits}

tf.saved_model.simple_save(sess, RUTA_SALIDA, inputs, outputs)

# Guardamos la salida antes de cerrar la session actual
output_red = sess.run(logits, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_red_session_entrenamiento.txt", 'w') as file:
    file.write(str(output_red))


conv1_np = sess.run(conv1, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_conv1.txt", 'w') as file:
    file.write(str(conv1_np))

conv1r_np = sess.run(conv1r, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_conv1r.txt", 'w') as file:
    file.write(str(conv1r_np))

conv2_np = sess.run(conv2, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_conv2.txt", 'w') as file:
    file.write(str(conv2_np))

conv2r_np = sess.run(conv2r, feed_dict={x: np.expand_dims(mnist.test.images[0,:], axis=0)})
with open(RUTA_SALIDA+"output_conv2r.txt", 'w') as file:
    file.write(str(conv2r_np))
