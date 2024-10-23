import time
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def booth_function(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

def calculate_booth_minimum(beta, iterations):
    grad_booth = grad(booth_function)
    x = np.array([0.0, 0.0])
    history = []
    history.append(x.copy())
    for i in range(iterations):
        grad_val = grad_booth(x)
       # x = x - beta * grad_val # new x = old x - beta * grad val (moving with gradient descend)
        x = np.clip(x - beta * grad_val, -10, 10)
        history.append(x.copy())
    return history

def draw_plot(beta, iterations):
    start_time = time.process_time()
    history = calculate_booth_minimum(beta, iterations)
    end_time = time.process_time()
    X, Y = np.meshgrid(np.arange(-10, 10, 0.1), np.arange(-10, 10, 0.1))
    Z = np.empty(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = booth_function(np.array([X[i, j], Y[i, j]]))
    plt.contour(X, Y, Z, 50)

    for i in range(1, len(history)):
        plt.arrow(history[i - 1][0], history[i - 1][1],
                  history[i][0] - history[i - 1][0], history[i][1] - history[i - 1][1],
                  head_width=0.3, head_length=0.3, fc='r', ec='r')

    plt.title("Gradient Descent dla funkcji Bootha \nCzas: " + "{0:02f}s".format(end_time - start_time))
    plt.savefig(f"booth/booth_plot_beta{beta}.png")

    plt.close()
    with open("zadanie2_booth.csv", "a") as file:
        file.write(f"{beta}," + "{0:02f}s".format(end_time - start_time) + f",{history[-1]};\n")

if __name__ == '__main__':
    with open("zadanie2_booth.csv", "w") as file:
        file.write("beta,time;\n")
    beta_list = [ 1, 0.1, 0.05, 0.025, 0.01, 0.001, 0.0001]

    for i in beta_list:
        draw_plot(i, 100)
