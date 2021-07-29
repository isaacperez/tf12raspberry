import os
import numpy as np
np.set_printoptions(threshold=np.nan)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import tag_constants
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

RUTA_SALIDA = './modelo_entrenado_mnist_cnn9/'
RUTA_SALIDA_C = 'modelo_entrenado_mnist_cnn9/'

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

    wc2 = tf.get_default_graph().get_tensor_by_name('W_c2:0')
    bc2 = tf.get_default_graph().get_tensor_by_name('b_c2:0')

    wc3 = tf.get_default_graph().get_tensor_by_name('W_c3:0')
    bc3 = tf.get_default_graph().get_tensor_by_name('b_c3:0')

    wc4 = tf.get_default_graph().get_tensor_by_name('W_c4:0')
    bc4 = tf.get_default_graph().get_tensor_by_name('b_c4:0')

    wc5 = tf.get_default_graph().get_tensor_by_name('W_c5:0')
    bc5 = tf.get_default_graph().get_tensor_by_name('b_c5:0')

    wn1 = tf.get_default_graph().get_tensor_by_name('W_n1:0')
    bn1 = tf.get_default_graph().get_tensor_by_name('b_n1:0')

    logits = tf.get_default_graph().get_tensor_by_name('logits:0')
    """
    conv2r = tf.get_default_graph().get_tensor_by_name('conv2_con_activacion:0')
    conv1r = tf.get_default_graph().get_tensor_by_name('conv1_con_activacion:0')
    conv4 = tf.get_default_graph().get_tensor_by_name('conv4:0')
    conv2 = tf.get_default_graph().get_tensor_by_name('conv2_con_bias:0')
    maxPool = tf.get_default_graph().get_tensor_by_name('maxPool:0')

    wc4_np = sess.run(wc4)
    contador = 0
    #for i in range(8):
    a = sess.run(tf.nn.conv2d(conv2r[:,:,:,:], tf.expand_dims(wc4[:,:,:,0], axis=-1), strides=[1, 2, 2, 1], padding='SAME'),feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)})
    a = a[:,0,0,:]
    print("Resultado conv:", a, "\n")

    contador = 0
    for i in range(8):
        print("Canal de entrada", i)
        print("\tPesos:", wc4_np[:,:,i,0])
        #conv2r_np, a = sess.run([conv2r, tf.nn.conv2d(conv2r[:,:,:,:], tf.expand_dims(tf.expand_dims(wc4[:,:,i,0], axis=-1)), strides=[1, 2, 2, 1], padding='SAME')],feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)})
        #a = a[:,0,0,i]
        #contador += a
        conv2r_np = sess.run(conv2r, feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)})
        print("\tEntrada:", conv2r_np[0,[0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2],i])
        a = sess.run(tf.nn.conv2d(tf.expand_dims(conv2r[:,:,:,i], axis=-1), tf.expand_dims(tf.expand_dims(wc4[:,:,i,0], axis=-1), axis=-1), strides=[1, 2, 2, 1], padding='SAME'),feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)})

        print("\tResultado conv:", a[:,0,0,:], "\n")
        contador += a[:,0,0,:]
    print("\nSuma total:", contador)

    print(a[:,0,0,:])

    #print(sess.run(conv2, feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)}))
    #print(conv2r_np)
    print(sess.run(maxPool, feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)}))
    print(sess.run(conv1r, feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)})[0,[8,8,9,9],[6,7,6,7],1])
    """


    # Después obtenemos la y que da la red sobre estas 10 imágenes y guardamos el resutlado esperado
    output_red = sess.run(logits, feed_dict={X: np.expand_dims(mnist.test.images[0,:], axis=0)}) # (10, 10) (batch, neuronas)

    with open(RUTA_SALIDA+"output_red_session_cargada_disco.txt", 'w') as file:
        file.write(str(output_red))


    def guardarFiltroEnFormatoC(shapeFiltroActual, wc_np, bc_np):

        with open("./ficheros_c/"+RUTA_SALIDA_C+"wc"+strFiltroActual+".h", 'w') as file:
            string = "float wc"+strFiltroActual+"[] = {";
            for canalEntrada in range(shapeFiltroActual[0]):
                for fila in range(shapeFiltroActual[1]):
                    for col in range(shapeFiltroActual[2]):
                        for canalFiltro in range(shapeFiltroActual[3]):
                            string += str(wc_np[fila,col, canalEntrada, canalFiltro]) + ", "

            string = string[:-4]
            file.write(string + "};")

        with open("./ficheros_c/"+RUTA_SALIDA_C+"bc"+strFiltroActual+".h", 'w') as file:
            string = "float bc"+strFiltroActual+"[] = {";
            for k in range(shapeFiltroActual[3]):
                string += str(bc_np[k])
                if k<(shapeFiltroActual[3]-1):
                    string += ", "
            file.write(string + "};")


    # CONV 1
    strFiltroActual = "1"
    shapeFiltroActual = [1,1,1,4] # CanalesEntrada, FilasFiltro, ColumnasFiltro, CanalesFiltro
    wc_np, bc_np = sess.run([wc1, bc1])
    guardarFiltroEnFormatoC(shapeFiltroActual, wc_np, bc_np)

    # CONV 2
    strFiltroActual = "2"
    shapeFiltroActual = [4,3,3,8] # CanalesEntrada, FilasFiltro, ColumnasFiltro, CanalesFiltro
    wc_np, bc_np = sess.run([wc2, bc2])
    guardarFiltroEnFormatoC(shapeFiltroActual, wc_np, bc_np)

    # CONV 3
    strFiltroActual = "3"
    shapeFiltroActual = [4,1,1,8] # CanalesEntrada, FilasFiltro, ColumnasFiltro, CanalesFiltro
    wc_np, bc_np = sess.run([wc3, bc3])
    guardarFiltroEnFormatoC(shapeFiltroActual, wc_np, bc_np)

    # CONV 4
    strFiltroActual = "4"
    shapeFiltroActual = [8,3,3,12] # CanalesEntrada, FilasFiltro, ColumnasFiltro, CanalesFiltro
    wc_np, bc_np = sess.run([wc4, bc4])
    guardarFiltroEnFormatoC(shapeFiltroActual, wc_np, bc_np)

    # CONV 5
    strFiltroActual = "5"
    shapeFiltroActual = [8,1,1,12] # CanalesEntrada, FilasFiltro, ColumnasFiltro, CanalesFiltro
    wc_np, bc_np = sess.run([wc5, bc5])
    guardarFiltroEnFormatoC(shapeFiltroActual, wc_np, bc_np)

    # NN
    wn1_np, bn1_np = sess.run([wn1, bn1])
    with open("./ficheros_c/"+RUTA_SALIDA_C+"wn1.h", 'w') as file:
        NUM_CONEXIONES_POR_NEURONA = 12
        string = "float wn1[] = {";
        for k in range(10):
            for j in range(NUM_CONEXIONES_POR_NEURONA):
                string += str(wn1_np[j,k])
                if k<9 or (k==9 and j <NUM_CONEXIONES_POR_NEURONA-1):
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
