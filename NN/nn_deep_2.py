import numpy as np
import matplotlib.pyplot as plt
import math
from NN.neural_network_tut import load_data


def load_dataset():
    X_train, Y_train, X_test, Y_test = load_data()

    X_train, X_test = X_train.T / 255, X_test.T / 255
    Y_train, Y_test = Y_train.reshape((1, Y_train.shape[0])), Y_test.reshape((1, Y_test.shape[0]))

    return X_train, Y_train, X_test, Y_test


def one_hot_encoding(Y, num_labels):
    one_hot = np.zeros((num_labels, Y.shape[1]))

    for i, val in enumerate(Y[0]):
        one_hot[val, i] = 1.0

    return one_hot


def sigmoid(z):
    return 1 / (1 + np.exp(-z)), z


def sigmoid_backward(dA, activation_cache):
    Z = activation_cache
    a, _ = sigmoid(Z)

    dZ = dA * (a * (1 - a))
    return dZ


def initialize_weights(layer_dims):
    L = len(layer_dims)
    parameters = {}

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * \
                                   np.sqrt(2 / (layer_dims[l] + layer_dims[l-1]))
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


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


def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l + 1)] = np.zeros(parameters["b" + str(l+1)].shape)

    return v, s


def linear_forward_unit(A, W, b):
    Z = np.dot(W, A) + b
    linear_cache = (A, W, b)

    return Z, linear_cache


def linear_activation_unit(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward_unit(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    caches = (linear_cache, activation_cache)
    return A, caches


def feed_forward_propagation(X, parameters):
    L = len(parameters) // 2
    m = X.shape[1]
    caches = []
    A = X

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_unit(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], "sigmoid")
        caches.append(cache)

    AL, cache = linear_activation_unit(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    m = AL.shape[1]

    logpreds = Y * np.log(AL) + (1 - Y) * np.log(AL)
    error = (-1/m) * np.sum(logpreds)
    return error


def linear_backward_unit(dZ, linear_cache):
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (-1/m) * np.dot(dZ, A_prev.T)
    db = (-1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backward_activation_unit(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward_unit(dZ, linear_cache)

    return dA_prev, dW, db


def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]

    # Initialize backpropagation
    AL = np.clip(AL, 1e-99, 1)  # clip to avoid 0
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Calculate fro the Lth layer
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = backward_activation_unit(dAL, current_cache, "sigmoid")

    # For layers l = L-2...0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backward_activation_unit(grads["dA" + str(l+1)], current_cache, "sigmoid")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


def update_parameters_adam(parameters, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l+1)] = beta1 * v["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta1 * v["db" + str(l+1)] + (1 - beta1) * grads["db" + str(l+1)]

        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / (1 - beta1**t)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / (1 - beta1**t)

        s["dW" + str(l+1)] = beta2 * s["dW" + str(l+1)] + (1 - beta1) * grads["dW" + str(l+1)]**2
        s["db" + str(l+1)] = beta2 * s["db" + str(l+1)] + (1 - beta2) * grads["db" + str(l+1)]**2

        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / (1 - beta2**t)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / (1 - beta2**t)

        # Update parameters
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected["dW" + str(l+1)] / (np.sqrt(s_corrected["dW" + str(l+1)]) + epsilon)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * v_corrected["db" + str(l + 1)] / (
                    np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s


def predict(X, Y, parameters):
    m = X.shape[1]
    Y_enc = one_hot_encoding(Y, 10)
    p = np.zeros(Y_enc.shape)

    AL, caches = feed_forward_propagation(X, parameters)
    max_indices = np.argmax(AL, axis=0)
    p[max_indices, :] = 1.0

    print(f"Accuracy: {np.mean(p == Y_enc)}")

    return p



def test_set(X, Y, preds):
    Y_enc = one_hot_encoding(Y, 10)

    miscl = np.where(Y_enc != preds)[1]
    print(miscl, miscl.shape)
    miss_img = X[:, miscl[:25]]
    correct_lab = Y[:, miscl[:25]]
    miss_lab = preds[:, miscl[:25]]

    # fig, ax = plt.subplots(5, 5, sharex=True, sharey=True)
    # ax = ax.flatten()
    #
    # for i in range(25):
    #     img = miss_img[:, i].reshape(28, 28)
    #     ax[i].imshow(img, cmap="Greys", interpolation='nearest')
    #     ax[i].set_title(f'{i+1} t: {correct_lab} p: {miss_lab}')
    #
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # plt.tight_layout()
    # plt.show()


def model(X, Y, layer_dims, mini_batch_size=64, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=20, print_cost=True):
    L = len(layer_dims)
    costs = []
    t = 0
    m = X.shape[1]

    parameters = initialize_weights(layer_dims)
    v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        minibatches = random_mini_batches(X, Y, mini_batch_size)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch

            minibatch_Y = one_hot_encoding(minibatch_Y, 10)

            # Forward Propagation
            AL, caches = feed_forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost_total += compute_cost(AL, minibatch_Y)

            # Backward Propagation
            grads = backward_propagation(AL, minibatch_Y, caches)

            # Update parameters
            t = t + 1
            parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        cost_avg = cost_total / m

        if print_cost and i % 1 == 0:
            print(f"Cost at iteration {i}: {cost_avg}")
        if print_cost and i % 1 == 0:
            costs.append(cost_avg)

    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("iteration (per epoch)")
    plt.title(f"Learning Rate: {learning_rate}")
    plt.show()

    return parameters


X_train, Y_train, X_test, Y_test = load_dataset()

parameters = model(X_train, Y_train, [X_train.shape[0], 15, 15, 10])
print("Training Set: ", end="")
predict(X_train, Y_train, parameters)
print("Test Set: ", end="")
pred_test = predict(X_test, Y_test, parameters)
print(np.where(one_hot_encoding(Y_test,10) != pred_test)[1].shape)
# test_set(X_test, Y_test, pred_test)








