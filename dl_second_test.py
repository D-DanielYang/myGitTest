import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from dl_basic_function import load_dataset
import math

# region Description # Input data and print something
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_num = train_set_x_orig.shape[0]
test_num = test_set_x_orig.shape[0]
pix_num = train_set_x_orig.shape[1]
chanel_num = train_set_x_orig.shape[3]
print('\n' + '---------------Input information---------------' + '\n')
print('train_set_x_orig.shape = ' + str(train_set_x_orig.shape))
print('train_set_y.shape = ' + str(train_set_y.shape))
print('test_set_x_orig.shape = ' + str(test_set_x_orig.shape))
print('test_set_y.shape = ' + str(test_set_y.shape))
print('train_num = ' + str(train_num))
print('test_num = ' + str(test_num))
print('pix_num = ' + str(pix_num))
print('chanel_num = ' + str(chanel_num))

# Reshape data
train_set_x_flat = train_set_x_orig.reshape(train_num, -1).T
test_set_x_flat = test_set_x_orig.reshape(test_num, -1).T
print('\n' + '---------------After reshaping---------------' + '\n')
print('train_set_x_flat = ' + str(train_set_x_flat.shape))
print('test_set_x_flat = ' + str(train_set_x_flat.shape))

# Standarize data
train_set_x = train_set_x_flat / 255.0
test_set_x = test_set_x_flat / 255.0
print('\n' + '---------------After Standaring---------------' + '\n')
print('Check for traindata = ' + str(train_set_x[0:5, 0]))
print('Check for testdata = ' + str(test_set_x[0:5, 0]))
# endregion


# region Description //some usefull function
def sigmoid(Z):
    s = 1.0 / (1.0 + np.exp(-Z))
    return s


def sigmoid_prime(Z):
    s = sigmoid(Z) * (1 - sigmoid(Z))
    return s


def tanh(x):
    s = (np.exp(x) - np.exp(-x)) / ((np.exp(x) + np.exp(-x)))
    return s


def tanh_prime(x):
    s = 1 - np.power(tanh(x), 2)
    return s


def relu(Z):
    s = np.maximum(0, Z)
    return s


def relu_prime(Z):
    s = Z
    s[Z <= 0] = 0
    s[Z > 0] = 1
    return s


def initial_weights(layer):
    # layer :[nx, n1, n2, ... nL]
    np.random.seed(2)
    L = len(layer)
    parameters = {}
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer[l], layer[l-1])*np.sqrt(2.0/layer[l-1])
        parameters["b"+str(l)] = np.zeros((layer[l],1))
    return parameters
# endregion   %%


def train(X, Y, layer, learning_rate=0.0075, itr_num=3000, isprint=True):
    '''''
    Input:
    X: training dataset with shape(nx,m)
    Y: training dataset with shape(1,m)
    learning_rate: default with 0.0075
    itr_num: max number of iteration
    isprint: print cost(True) or not(False)

    Output:
    parameters : dict data{"W1","b1",...,"WL","bL"},include final weights
    '''''
    m = Y.shape[1]  # number of training set
    Fcahe = {}  # Save forward propagation variable(Z,A)
    Bcahe = {}  # Save backward propagation variable(dZ,dW,db)
    costs = []  # Save cost series to plot

    Fcahe["A" + str(0)] = X  # Set X as A0
    parameters = initial_weights(layer)  # Initial weights
    L = len(parameters) // 2  # Layers of neural netwoek

    for itr in range(itr_num):

        # Forward propagation
        for l in range(1, L):
            Fcahe["Z" + str(l)] = np.dot(parameters["W" + str(l)], Fcahe["A" + str(l - 1)]) + parameters["b" + str(l)]
            Fcahe["A" + str(l)] = relu(Fcahe["Z" + str(l)])
        Fcahe["Z" + str(L)] = np.dot(parameters["W" + str(L)], Fcahe["A" + str(L - 1)]) + parameters["b" + str(L)]
        Fcahe["A" + str(L)] = sigmoid(Fcahe["Z" + str(L)])
        cost = -(1.0 / m) * np.sum(Y * np.log(Fcahe["A" + str(L)]) + (1 - Y) * np.log(1 - Fcahe["A" + str(L)]))
        cost = np.squeeze(cost)

        # Backward propagation
        Bcahe["dZ" + str(L)] = (-(
                    np.divide(Y, Fcahe["A" + str(L)]) - np.divide(1 - Y, 1 - Fcahe["A" + str(L)]))) * sigmoid_prime(
            Fcahe["Z" + str(L)])
        Bcahe["dW" + str(L)] = (1.0 / m) * np.dot(Bcahe["dZ" + str(L)], Fcahe["A" + str(L - 1)].T)
        Bcahe["db" + str(L)] = (1.0 / m) * np.sum(Bcahe["dZ" + str(L)], axis=1, keepdims=True)
        for l in reversed(range(L - 1)):
            Bcahe["dZ" + str(l + 1)] = np.dot(parameters["W" + str(l + 2)].T, Bcahe["dZ" + str(l + 2)]) * relu_prime(
                Fcahe["Z" + str(l + 1)])
            Bcahe["dW" + str(l + 1)] = (1.0 / m) * np.dot(Bcahe["dZ" + str(l + 1)], Fcahe["A" + str(l)].T)
            Bcahe["db" + str(l + 1)] = (1.0 / m) * np.sum(Bcahe["dZ" + str(l + 1)], axis=1, keepdims=True)

        # Update weights
        for l in range(L):
            parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * Bcahe["dW" + str(l + 1)]
            parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * Bcahe["db" + str(l + 1)]

        if isprint and itr % 200 == 0:
            print('cost after ' + str(itr) + ' iteration is :' + str(cost))
            costs.append(cost)

    # Plot the cost  curve vary with itr_num
    plt.plot(np.squeeze(costs))
    plt.xlabel("itr_num(per hundred)")
    plt.ylabel("cost")
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return parameters


def predict(X, Y, parameters):
    '''''
    Input:
    X: test dataset with shape(nx,m)
    Y: test dataset with shape(1,m)
    parameters : dict data{"W1","b1",...,"WL","bL"}

    Output: 
    pred: predicted labels of test dataset

    '''''
    num = X.shape[1]
    pred = np.zeros((1, num))
    L = len(parameters) // 2
    A0 = X
    for l in range(1, L):
        Z = np.dot(parameters["W" + str(l)], A0) + parameters["b" + str(l)]
        A = relu(Z)
        A0 = A
    Z = np.dot(parameters["W" + str(L)], A0) + parameters["b" + str(L)]
    A = sigmoid(Z)
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5:
            pred[0, i] = 0
        else:
            pred[0, i] = 1
    print("accuracy:{}%".format(100 - np.mean(np.abs(pred - Y)) * 100))
    return pred


parameters = train(train_set_x, train_set_y, [12288, 1], learning_rate=0.01, itr_num=3000)
test_pred = predict(test_set_x, test_set_y, parameters)

'''
parameters = train(train_set_x, train_set_y, [12288, 5, 1], learning_rate=0.01, itr_num=3000)
test_pred = predict(test_set_x, test_set_y, parameters)

parameters = train(train_set_x, train_set_y, [12288, 20, 5, 1], learning_rate=0.01, itr_num=3000)
test_pred = predict(test_set_x, test_set_y, parameters)
'''
