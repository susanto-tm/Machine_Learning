# In[]:

import numpy as np
import struct
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


# In[]:

def load_data():
    with open('mnist/train-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack(">II", labels.read(8))
        train_labels = np.fromfile(labels, dtype=np.uint8)
    with open("mnist/train-images.idx3-ubyte", 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack(">IIII", imgs.read(16))
        train_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)
    with open('mnist/t10k-labels.idx1-ubyte', 'rb') as labels:
        magic, n = struct.unpack(">II", labels.read(8))
        test_labels = np.fromfile(labels, dtype=np.uint8)
    with open("mnist/t10k-images.idx3-ubyte", 'rb') as imgs:
        magic, num, nrows, ncols = struct.unpack(">IIII", imgs.read(16))
        test_images = np.fromfile(imgs, dtype=np.uint8).reshape(num, 784)

    return train_images, train_labels, test_images, test_labels

# In[]:

def load_dataset():
    X_train, Y_train, X_test, Y_test = load_data()

    X_train, X_test = X_train.T / 255, X_test.T / 255
    Y_train, Y_test = Y_train.reshape((1, Y_train.shape[0])), Y_test.reshape((1, Y_test.shape[0]))

    return X_train, Y_train, X_test, Y_test

# In[]:

X_train, Y_train, X_test, Y_test = load_dataset()

print(X_train.shape, X_test.shape)

/