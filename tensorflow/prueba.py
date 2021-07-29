import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

a = mnist.test.images[0,:].reshape(28,28)

string = ""
for i in range(28):
    for j in range(28):
        string += str(a[i,j]) + " "
    string += "\n"

print(string)
