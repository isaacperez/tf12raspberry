import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

import tensorflow as tf

# define a neural network (softmax logistic regression)
x = tf.placeholder(tf.float32, [None, 784], name="X")
W = tf.Variable(tf.zeros([784, 10]), name="W")
b = tf.Variable(tf.zeros([10]), name="B")
logits = tf.matmul(x, W) + b
logits = tf.identity(logits, name="logits")
y = tf.nn.softmax(logits) # the equation

inputs = {"X": x}
outputs = {"W": W,
           "B": b,
           "logits": logits}

# define the train step to minimize the cross entropy with SGD
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

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
tf.saved_model.simple_save(sess, './modelo_entrenado_mnist_fc/', inputs, outputs)

# Guardamos la salida antes de cerrar la session actual
output_red = sess.run(logits, feed_dict={x: mnist.test.images[:10,:]})
with open("./modelo_entrenado_mnist_fc/output_red_session_entrenamiento.txt", 'w') as file:
    file.write(str(output_red))
