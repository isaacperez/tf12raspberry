import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import tag_constants
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
"""
# Una vez entrenado cogemos las 10 primeras imágenes de test y las guardamos en el formato de C esperado
for i in range(10):
    with open("./imagenes_de_test/imagen_" + str(i) + "_de_test.h", 'w') as file:
        string = "float img[] = {";
        for j in range(784):
            string += str(mnist.test.images[i,j])
            if j<783:
                string += ", "

        file.write(string + "};")



# Cargamos la red entrenada
with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tag_constants.SERVING],
    './modelo_entrenado_mnist_fc/',
    )

    X = tf.get_default_graph().get_tensor_by_name('X:0')
    W = tf.get_default_graph().get_tensor_by_name('W:0')
    b = tf.get_default_graph().get_tensor_by_name('B:0')
    logits = tf.get_default_graph().get_tensor_by_name('logits:0')

    # Después obtenemos la y que da la red sobre estas 10 imágenes y guardamos el resutlado esperado
    output_red = sess.run(logits, feed_dict={X: mnist.test.images[:10,:]}) # (10, 10) (batch, neuronas)

    with open("./modelo_entrenado_mnist_fc/output_red_session_cargada_disco.txt", 'w') as file:
        file.write(str(output_red))


    # Por último guardamos W y b en el formato esperado de C
    W_np, b_np = sess.run([W, b])
    with open("./ficheros_c/mnist_fc/w.h", 'w') as file:
        string = "float W[] = {";
        for k in range(10):
            for j in range(784):
                string += str(W_np[j,k])
                if k<9 or (k==9 and j <783):
                    string += ", "
        file.write(string + "};")

    with open("./ficheros_c/mnist_fc/b.h", 'w') as file:
        string = "float B[] = {";
        for k in range(10):
            string += str(b_np[k])
            if k<9:
                string += ", "
        file.write(string + "};")

    #print(W_np.shape, b.shape) (784,10) (10,)

"""
"""
with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tag_constants.SERVING],
    './modelo_entrenado_mnist_fc/',
    )

    X = tf.get_default_graph().get_tensor_by_name('X:0')
    W = tf.get_default_graph().get_tensor_by_name('W:0')
    b = tf.get_default_graph().get_tensor_by_name('B:0')

    # Después obtenemos la y que da la red sobre estas 10 imágenes y guardamos el resutlado esperado
    output_red = sess.run(tf.matmul(X, W), feed_dict={X: mnist.test.images[:10,:]}) # (10, 10) (batch, neuronas)

    print(output_red)
"""
with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tag_constants.SERVING],
    './modelo_entrenado_mnist_fc/',
    )

    X = tf.get_default_graph().get_tensor_by_name('X:0')
    W = tf.get_default_graph().get_tensor_by_name('W:0')
    b = tf.get_default_graph().get_tensor_by_name('B:0')

    # Después obtenemos la y que da la red sobre estas 10 imágenes y guardamos el resutlado esperado
    W_np = sess.run(W)
    with open("./modelo_entrenado_mnist_fc/w_0.txt", 'w') as file:
        file.write(str(W_np[:,0]))
    with open("./modelo_entrenado_mnist_fc/w_1.txt", 'w') as file:
        file.write(str(W_np[:,1]))
