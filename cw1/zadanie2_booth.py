import numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def booth_function(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2

grad_booth = grad(booth_function)

beta = 0.1
x = np.array([50.0, 20.0])
iterations = 100

history = []

# Gradient Descent
for i in range(iterations):
    grad_val = grad_booth(x)
    x = x - beta * grad_val  # aktualizacja punktu
    history.append(x.copy())  # zapisz punkt do historii

# Rysowanie poziomic i kroków algorytmu
X, Y = np.meshgrid(np.arange(-100, 100, 0.1), np.arange(-100, 100, 0.1))
Z = (X + 2 * Y - 7) ** 2 + (2 * X + Y - 5) ** 2
plt.contour(X, Y, Z, levels=100)  # poziomice funkcji

# Rysowanie strzałek przedstawiających kroki algorytmu
for i in range(1, len(history)):
    plt.arrow(history[i-1][0], history[i-1][1],
              history[i][0] - history[i-1][0], history[i][1] - history[i-1][1],
              head_width=0.3, head_length=0.3, fc='k', ec='k')

plt.title("Gradient Descent dla funkcji Bootha")
plt.show()
plt.savefig()