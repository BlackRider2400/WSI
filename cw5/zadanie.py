#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 16:51:50 2021

@author: Rafał Biedrzycki
Kodu tego mogą używać moi studenci na ćwiczeniach z przedmiotu Wstęp do Sztucznej Inteligencji.
Kod ten powstał aby przyspieszyć i ułatwić pracę studentów, aby mogli skupić się na algorytmach sztucznej inteligencji.
Kod nie jest wzorem dobrej jakości programowania w Pythonie, nie jest również wzorem programowania obiektowego, może zawierać błędy.

Nie ma obowiązku używania tego kodu.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [8, 2]

L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x * np.sqrt(p[0] + 1)) + np.cos(x * np.sqrt(p[1] + 1))


x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

np.random.seed(1)


# f logistyczna jako przykład sigmoidalej
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# pochodna fun. 'sigmoid'
def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)


# f. straty
def nloss(y_out, y):
    return (y_out - y) ** 2


# pochodna f. straty
def d_nloss(y_out, y):
    return 2 * (y_out - y)


class DlNet:
    def __init__(self, x, y, hidden_layer_size=9):
        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.y_out = np.zeros_like(y).reshape(-1, 1)

        self.HIDDEN_L_SIZE = 100
        self.LR = 0.003

        self.W1 = np.random.randn(self.x.shape[1], self.HIDDEN_L_SIZE)
        self.b1 = np.zeros((1, self.HIDDEN_L_SIZE))

        self.W2 = np.random.randn(self.HIDDEN_L_SIZE, 1)
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_out = self.z2

    def predict(self, x):
        x = x.reshape(-1, 1)
        self.forward(x)
        return self.y_out

    def backward(self, x, y):
        m = y.shape[0]

        gradient_loss_to_output = d_nloss(self.y_out, y)

        gradient_weights_output_layer = np.dot(self.a1.T, gradient_loss_to_output) / m
        gradient_bias_output_layer = np.sum(gradient_loss_to_output, axis=0, keepdims=True) / m

        gradient_loss_to_hidden_activation = np.dot(gradient_loss_to_output, self.W2.T)
        gradient_loss_to_hidden_sum = gradient_loss_to_hidden_activation * d_sigmoid(self.z1)

        gradient_weights_hidden_layer = np.dot(x.T, gradient_loss_to_hidden_sum) / m
        gradient_bias_hidden_layer = np.sum(gradient_loss_to_hidden_sum, axis=0, keepdims=True) / m

        self.W2 -= self.LR * gradient_weights_output_layer
        self.b2 -= self.LR * gradient_bias_output_layer
        self.W1 -= self.LR * gradient_weights_hidden_layer
        self.b1 -= self.LR * gradient_bias_hidden_layer

    def train(self, x_set, y_set, iters):
        for i in range(iters):
            self.forward(x_set)
            self.backward(x_set, y_set)

            if i % 1000 == 0:
                loss = np.mean(nloss(self.y_out, y_set))
                print(f"Iteracja {i}, Strata: {loss}")


hidden_layer_sizes = [9, 25, 50]
training_iterations = [ 15000, 150000,1500000]
results_dir = "results"

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

for iters in training_iterations:
    mse_values = []
    print(f"\nTestowanie dla {iters} iteracji trenowania:")
    for size in hidden_layer_sizes:
        print(f"  Trenowanie sieci z {size} neuronami w warstwie ukrytej...")

        nn = DlNet(x, y, hidden_layer_size=size)
        nn.train(nn.x, nn.y, iters)

        predictions = nn.predict(x)
        mse = np.mean((y - predictions.flatten()) ** 2)
        mse_values.append(mse)

        plt.figure()
        plt.plot(x, y, 'r', label='Oryginalna funkcja J(x)')
        plt.plot(x, predictions, 'b', label=f'Aproksymacja ({size} neuronów, {iters} iteracji)')
        plt.legend()
        plt.title(f'Liczba neuronów: {size}, Iteracje: {iters}')
        plt.savefig(f"{results_dir}/approximation_{size}_neurons_{iters}_iters.png")
        plt.close()

    plt.figure()
    plt.plot(hidden_layer_sizes, mse_values, 'o-', label=f'MSE ({iters} iteracji)')
    plt.xlabel('Liczba neuronów w warstwie ukrytej')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title(f'Jakość aproksymacji w zależności od liczby neuronów (Iteracje: {iters})')
    plt.legend()
    plt.savefig(f"{results_dir}/mse_vs_neurons_{iters}_iters.png")
    plt.close()

