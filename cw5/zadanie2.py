import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
# ToDo tu prosze podac pierwsze cyfry numerow indeksow
p = [8, 2]

L_BOUND = -5
U_BOUND = 5


def q(x):
    return np.sin(x * np.sqrt(p[0] + 1)) + np.cos(x * np.sqrt(p[1] + 1))


x = np.linspace(L_BOUND, U_BOUND, 100)
y = q(x)

model = Sequential([
    Dense(units=25, activation='sigmoid', input_shape=[1]),
    Dense(units=1)
])
optimizer = SGD(learning_rate=0.005)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(x, y, epochs = 1500000)


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
    def __init__(self, x, y):
        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)
        self.y_out = np.zeros_like(y).reshape(-1, 1)

        self.HIDDEN_L_SIZE = 25
        self.LR = 0.005

        self.W1 = np.random.randn(self.x.shape[1], self.HIDDEN_L_SIZE)
        self.b1 = np.zeros((1, self.HIDDEN_L_SIZE))

        self.W2 = np.random.randn(self.HIDDEN_L_SIZE, 1)
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2

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


nn = DlNet(x, y)
nn.train(nn.x, nn.y, 1000000)

yh = nn.predict(x)
yn = model.predict(x)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.plot(x, y, 'r', label='Oryginalna funkcja J(x)')
plt.plot(x, yh, 'b', label='Aproksymacja sieci')
plt.plot(x, yn, 'g', label='aproksymacja tensorflow')
plt.legend()
plt.savefig('tensor.png')