import time
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from cec2017.functions import f1


def calculate_cec_minimum(beta, iterations):
    grad_cec = grad(f1)
    x = np.array([0.0] * 10)
    history = []
    history.append(x.copy())

    for i in range(iterations):
        grad_val = grad_cec(x)
        x = np.clip(x - beta * grad_val, -100, 100)
        history.append(x.copy())

    return history


def draw_plot_cec(beta, iterations):
    start_time = time.process_time()
    history = calculate_cec_minimum(beta, iterations)
    end_time = time.process_time()

    X, Y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f1(np.array([X[i, j], Y[i, j]]))

    plt.contour(X, Y, Z, 50)

    for i in range(1, len(history)):
        plt.arrow(history[i - 1][0], history[i - 1][1],
                  history[i][0] - history[i - 1][0], history[i][1] - history[i - 1][1],
                  head_width=3, head_length=3, fc='r', ec='r')

    plt.title(f"Gradient Descent for CEC Function (beta={beta}) \nTime: " + "{0:02f}s".format(end_time - start_time))
    plt.savefig(f"cec/cec_plot_beta{beta}.png")
    plt.close()

    with open("zadanie2_cec.csv", "a") as file:
        file.write(f"{beta}," + "{0:02f}s".format(end_time - start_time) + f",{history[-1]};\n")


if __name__ == '__main__':
    dimensionality = 10
    with open("zadanie2_cec.csv", "w") as file:
        file.write("beta,time,last_position;\n")

    beta_list = [1, 0.1, 0.05, 0.025, 0.01, 0.001, 0.0001, 0.0000001, 0.0000000000001, 0.000000000000000001]

    for beta in beta_list:
        draw_plot_cec(beta, 100)
