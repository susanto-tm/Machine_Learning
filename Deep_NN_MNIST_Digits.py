#!/usr/bin/env python
# coding: utf-8

# In[1]:

import struct
import numpy as np
import matplotlib.pyplot as plt
import math


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


# In[2]:


def load_dataset():
    X_train, Y_train, X_test, Y_test = load_data()

    X_train, X_test = X_train.T / 255, X_test.T / 255
    Y_train, Y_test = Y_train.reshape((1, Y_train.shape[0])), Y_test.reshape((1, Y_test.shape[0]))

    return X_train, Y_train, X_test, Y_test


# In[3]:


def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]
    minibatches = []

    permutation = np.random.permutation(m)
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_batches = math.floor(m / mini_batch_size)
    for k in range(num_complete_batches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        minibatch = (mini_batch_X, mini_batch_Y)
        minibatches.append(minibatch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, mini_batch_size * num_complete_batches:]
        mini_batch_Y = shuffled_Y[:, mini_batch_size * num_complete_batches:]

        minibatch = (mini_batch_X, mini_batch_Y)
        minibatches.append(minibatch)

    return minibatches


# In[4]:


def initialize_weights(layer_dims):
    parameters = {}
    L = len(layer_dims)
    
    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters


# In[80]:


def initialize_momentum(parameters):
    L = len(parameters) // 2
    v = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v


# In[89]:


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] = np.zeros(parameters["b" + str(l+1)].shape)
        
    return v, s


# In[5]:


def one_hot_encoding(Y, num_labels):
    y_enc = np.zeros((num_labels, Y.shape[1]))
    
    for i, val in enumerate(Y[0]):
        y_enc[val, i] = 1.0
        
    return y_enc


# In[6]:


def one_hot_to_labels(one_hot):
    labs = np.zeros((1, one_hot.shape[1]))
    
    for i in range(one_hot.shape[1]):
        labs[:, i] = np.nonzero(one_hot[:, i])
        
    return labs


# In[7]:


def relu(z):
    return np.maximum(0, z)


# In[19]:


def relu_backward(dA, activation_cache):
    Z = activation_cache
    
    dZ = np.array(dA, copy=True)  # same size as Z and dA
    dZ[Z <= 0] = 0  # when Z <= 0, dZ = 0, else same as dA since dZ = dA * g'(Z)
    
    return dZ


# In[9]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[10]:


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    A = sigmoid(Z)
    
    return dA * A * (1 - A) 


# In[60]:


def linear_activation_unit(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    
    if activation == "relu":
        A = relu(Z)
    
    elif activation == "sigmoid":
        A = sigmoid(Z)
    
    return A, Z

# In[105]:


def forward_propagation(X, parameters):
    L = len(parameters) // 2
    forward_cache = []
    A = X
    
    for l in range(1, L):
        A_prev = A
        A, Z = linear_activation_unit(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        forward_cache.append(((A_prev, parameters["W" + str(l)], parameters["b" + str(l)]), Z))
        
    AL, Z = linear_activation_unit(A, parameters["W" + str(L)], parameters["b" + str(L)], "sigmoid")
    forward_cache.append(((A, parameters["W" + str(L)], parameters["b" + str(L)]), Z))
    
    return AL, forward_cache


# In[12]:


def compute_cost(AL, Y):
    m = AL.shape[1]
    
    logpreds = Y * np.log(AL) + (1 - Y) * np.log(1 - AL)
    error = (-1/m) * np.sum(logpreds)
    
    return error


# In[68]:


def backward_activation(dA, forward_cache, activation):
    linear_cache, Z = forward_cache
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]
    
    if activation == "relu":
        dZ = relu_backward(dA, Z)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, Z)
        
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[55]:


def backward_propagation(AL, Y, forward_cache, parameters):
    grads = {}
    L = len(forward_cache)
    
    # Initialize backpropagation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = forward_cache[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation(dAL, current_cache, "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = forward_cache[l]
        grads["dA" + str(l)], grads["dW" + str(l+1)], grads["db" + str(l+1)] = backward_activation(grads["dA" + str(l+1)], current_cache, "relu")
    
    return grads


# In[83]:


def update_parameters_gd(parameters, grads, learning_rate=0.01):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters


# In[77]:


def update_parameters_momentum(parameters, grads, v, learning_rate, beta1=0.9):
    L = len(parameters) // 2
    
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
    return parameters, v


# In[114]:


def update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]
        
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)
        
        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta2) * grads["dW" + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)]**2
        
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)
        
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected["db" + str(l+1)] / (np.sqrt(s_corrected["db" + str(l+1)]) + epsilon)
        
    return parameters, v, s
        


# In[64]:


def predict(X, Y, parameters):
    m = X.shape[1]
    y_enc = one_hot_encoding(Y, 10)
    p = np.zeros(y_enc.shape)
    
    AL, forward_cache = forward_propagation(X, parameters)
    pred_indices = np.argmax(AL, axis=0)
    
    for i in range(p.shape[1]):
        p[pred_indices[i], i] = 1.0
        
    p = one_hot_to_labels(p)
    
    print("Accuracy:", str(np.mean(p == Y)))
    
    return p
    


# In[138]:


def labeled_images(X, Y, preds, missed=False):
    fig, ax = plt.subplots(10, 10, sharex=True, sharey=True, figsize=(15, 15))
    ax = ax.flatten()
    
    if missed:
        mislbl = np.where(preds != Y)[1]
        mislbl_img = X[:, mislbl]
        mislbl_lbl = preds[:, mislbl]
        correct_lbl = Y[:, mislbl]
        print(f"Number of misclassified images:", mislbl.shape[0])
        
        for i in range(100):
            img = mislbl_img[:, i].reshape(28, 28)
            ax[i].imshow(img, cmap="Greys", interpolation='nearest')
            ax[i].set_title(f't: {correct_lbl[:, i]} p: {mislbl_lbl[:, i]}')
            
    else:
        for i in range(100):
            img = X_test[:, i].reshape(28, 28)
            ax[i].imshow(img, cmap="Greys", interpolation='nearest')
            ax[i].set_title(f't: {Y[:, i]} p: {preds[:, i]}')
    
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


# In[152]:


def model(X, Y, layer_dims, learning_rate=0.01, mini_batch_size=64, beta1=0.9, beta2=0.999, optimizer="gd", epoch=1000, epsilon=1e-8, print_cost=True):
    L = len(layer_dims)
    costs = []
    m = X.shape[1]
    t = 0
    
    parameters = initialize_weights(layer_dims)
    
    if optimizer == "momentum":
        v = initialize_momentum(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    for i in range(epoch):
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0
        
        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            minibatch_Y = one_hot_encoding(minibatch_Y, 10)
            
            # Feed forward propagation
            AL, forward_cache = forward_propagation(minibatch_X, parameters)
            
            # Compute Cost
            cost_total += compute_cost(AL, minibatch_Y)
            
            # Backpropagation
            grads = backward_propagation(AL, minibatch_Y, forward_cache, parameters)
            
            # Update Parameters
            if optimizer == "gd":
                parameters = update_parameters_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_momentum(parameters, grads, v, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate, epsilon=epsilon)
            
        cost_avg = cost_total / m    
        
        if print_cost and i % 1 == 0:
            print(f"Cost at iteration {i} is: {cost_avg}")
        if print_cost and i % 1 == 0:
            costs.append(cost_avg)
            
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Iteration (per 10 epoch)")
    plt.title(f"Learning Rate: {learning_rate}")
    plt.show()
    
    return parameters


# In[120]:


X_train, Y_train, X_test, Y_test = load_dataset()


# In[159]:


parameters = model(X_train, Y_train, [X_train.shape[0], 25, 25, 10], optimizer="adam", epsilon=1e-8, learning_rate=0.001, epoch=20)


# In[160]:


print("Training Set ", end="")
p = predict(X_train, Y_train, parameters)


# In[161]:


print("Test Set ", end="")
preds = predict(X_test, Y_test, parameters)


# In[162]:


labeled_images(X_test, Y_test, preds, missed=True)

