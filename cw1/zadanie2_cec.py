import time
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
from cec2017.functions import f1, f2, f3


def calculate_cec_minimum(beta, iterations, function):
    grad_cec = grad(function)
    x = np.array([0.0] * 10)
    history = []
    history.append(x.copy())

    for i in range(iterations):
        grad_val = grad_cec(x)
        x = np.clip(x - beta * grad_val, -100, 100)
        history.append(x.copy())

    return history


def draw_plot_cec(beta, iterations, function):
    start_time = time.process_time()
    history = calculate_cec_minimum(beta, iterations, function)
    end_time = time.process_time()

    X, Y = np.meshgrid(np.arange(-100, 100, 1), np.arange(-100, 100, 1))
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = function(np.array([X[i, j], Y[i, j]]))

    plt.contour(X, Y, Z, 50)

    for i in range(1, len(history)):
        plt.arrow(history[i - 1][0], history[i - 1][1],
                  history[i][0] - history[i - 1][0], history[i][1] - history[i - 1][1],
                  head_width=3, head_length=6, fc='r', ec='r')

    plt.title(f"Gradient Descent for CEC Function {function.__name__} (beta={beta}) \nTime: " + "{0:02f}s".format(end_time - start_time))
    plt.savefig(f"cec/cec{function.__name__}_plot_beta{beta}.png")
    plt.close()

    with open(f"zadanie2_cec{function.__name__}.csv", "a") as file:
        file.write(f"{beta}," + "{0:02f}".format(end_time - start_time) + "," +
                   np.array2string(history[-1], separator=' ', max_line_width=np.inf) + "\n")


if __name__ == '__main__':
    dimensionality = 10
    with open("zadanie2_cecf1.csv", "w") as file:
        file.write("beta,time,last_position\n")
    with open("zadanie2_cecf2.csv", "w") as file:
        file.write("beta,time,last_position\n")
    with open("zadanie2_cecf3.csv", "w") as file:
        file.write("beta,time,last_position\n")

    beta_list = [0.01]


    for i in range(13):
        beta_list.append(beta_list[-1] / 10)

    functions = [f1, f2, f3]

    for beta in beta_list:
        for f in functions:
            draw_plot_cec(beta, 1000, f)
