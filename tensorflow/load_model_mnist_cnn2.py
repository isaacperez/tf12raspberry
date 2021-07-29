import os
import numpy as np
np.set_printoptions(threshold=np.nan)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import tag_constants
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

RUTA_SALIDA = './modelo_entrenado_mnist_cnn2/'
RUTA_SALIDA_C = 'modelo_entrenado_mnist_cnn2/'

# Cargamos la red entrenada
with tf.Session() as sess:
    tf.saved_model.loader.load(
        sess,
        [tag_constants.SERVING],
        RUTA_SALIDA
    )

    X = tf.get_default_graph().get_tensor_by_name('X:0')

    wc1 = tf.get_default_graph().get_tensor_by_name('W_c1:0')
    bc1 = tf.get_default_graph().get_tensor_by_name('b_c1:0')

    wn1 = tf.get_default_graph().get_tensor_by_name('W_n1:0')
    bn1 = tf.get_default_graph().get_tensor_by_name('b_n1:0')

    logits = tf.get_default_graph().get_tensor_by_name('logits:0')

    # Después obtenemos la y que da la red sobre estas 10 imágenes y guardamos el resutlado esperado
    output_red = sess.run(logits, feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)}) # (10, 10) (batch, neuronas)

    with open(RUTA_SALIDA+"output_red_session_cargada_disco.txt", 'w') as file:
        file.write(str(output_red))


    # Por último guardamos W y b en el formato esperado de C
    wc1_np, bc1_np = sess.run([wc1, bc1])

    with open("./ficheros_c/"+RUTA_SALIDA_C+"wc1.h", 'w') as file:
        string = "float wc1[] = {";
        for canalEntrada in range(1):
            for fila in range(7):
                for col in range(7):
                    for canalFiltro in range(3):
                        string += str(wc1_np[fila,col, canalEntrada, canalFiltro]) + ", "

        string = string[:-4]
        file.write(string + "};")

    with open("./ficheros_c/"+RUTA_SALIDA_C+"bc1.h", 'w') as file:
        string = "float bc1[] = {";
        for k in range(3):
            string += str(bc1_np[k])
            if k<2:
                string += ", "
        file.write(string + "};")


    # Por último guardamos W y b en el formato esperado de C
    wn1_np, bn1_np = sess.run([wn1, bn1])
    with open("./ficheros_c/"+RUTA_SALIDA_C+"wn1.h", 'w') as file:
        string = "float wn1[] = {";
        for k in range(10):
            for j in range(147):
                string += str(wn1_np[j,k])
                if k<9 or (k==9 and j <147):
                    string += ", "

        string = string[:-4]
        file.write(string + "};")



    with open("./ficheros_c/"+RUTA_SALIDA_C+"bn1.h", 'w') as file:
        string = "float bn1[] = {";
        for k in range(10):
            string += str(bn1_np[k])
            if k<9:
                string += ", "
        file.write(string + "};")
