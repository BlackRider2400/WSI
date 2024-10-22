import numpy as np
from cec2017.functions import f1, f2, f3
import matplotlib.pyplot as plt
from autograd import grad


x = np.random.uniform(-100, 100, size=10)


value = f1(x)
print('q(x) = %.6f' % value)

grad_f1 = grad(f1)

beta = 0.001
iterations = 100
history = []

for i in range(iterations):
    grad_val = grad_f1(x)
    x = x - beta * grad_val  # aktualizacja punktu
    history.append(x.copy())



X = [h[0] for h in history]
Y = [h[1] for h in history]
plt.plot(X, Y, marker='o')
plt.title("Gradient Descent dla funkcji CEC f1 (2 wymiary)")
plt.show()
