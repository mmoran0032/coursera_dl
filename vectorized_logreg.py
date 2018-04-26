#!/usr/bin/env python3


import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import trange


def main():
    X, Y = generate_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    d = model(X_train, Y_train, X_test, Y_test)
    print(d)


def model(X_train, Y_train, X_test, Y_test,
          num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, _, costs = optimize(
        w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print('train accuracy: {.5f}%'.format(
        np.mean(np.abs(Y_prediction_train - Y_train))))
    print('test accuracy: {.5f}%'.format(
        np.mean(np.abs(Y_prediction_test - Y_test))))

    d = {
        'costs': costs,
        'Y_prediction_test': Y_prediction_test,
        'Y_prediction_train': Y_prediction_train,
        'w': w,
        'b': b,
        'learning_rate': learning_rate,
        'num_iterations': num_iterations
    }
    return d


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in trange(num_iterations, desc='propagate'):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f'Cost after iteration {i}: {cost}')

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(w.T @ X + b)
    cost = np.squeeze(
        -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A), axis=0) / m)
    dw = np.dot(X, (A - Y).T) / m
    db = (A - Y).sum() / m
    grads = {'dw': dw, 'db': db}
    return grads, cost


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(w.T @ X + b)
    for i in trange(A.shape[1], desc='predict'):
        Y_prediction[0, i] = (A[0, i] > 0.5).astype(int)
    return Y_prediction


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dimension):
    return np.zeros((dimension, 1)), 0.0


if __name__ == '__main__':
    main()
