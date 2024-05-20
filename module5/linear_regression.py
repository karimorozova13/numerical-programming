import numpy as np
import matplotlib.pyplot as plt

def y_hat(w, x_val):
    return w[1]*x_val + w[0]

def de(x, y, w, ind):
    m = len(x)
    error = [y_hat(w, x[i]) - y[i] for i in range(len(x))]
    if ind == 0:
        res = 1/m*(sum(error))
    if ind == 1:
        res = 1/m*(sum([x[i]*error[i] for i in range(len(x))]))
    return res

def gd2(x, y, w_0, iterations, gamma):
    w_i = w_0
    for i in range(iterations):
        w_p = w_i
        w_i = [w_p[0] - gamma*de(x, y, w_p, 0), w_p[1] - gamma*de(x, y, w_p, 1)]
        print('Iteration {} optimal w {}'.format(i, w_i))
    return w_i



# Згенеруємо деякі випадкові дані
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Ініціалізуємо параметри моделі
a, b = np.random.randn(), np.random.randn()

# Градієнтний спуск
learning_rate = 0.01
n_iterations = 1000

gd2(X, y, [a, b], n_iterations, learning_rate)
