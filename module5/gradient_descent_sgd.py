# Стохастичний градієнтний спуск (SGD)

import numpy as np
np.random.seed(42)


def linear_regression_sgd(X, y, learning_rate=0.1, epochs=1000):
    m, n = X.shape
    theta = np.random.randn(n, 1)  # ініціалізуємо випадковими значеннями параметри моделі

    for _ in range(epochs):

        # Обчислюємо градієнт відносно одного випадкового прикладу
        gradient = 2/m * X.T.dot(X.dot(theta) - y)

        # Оновлюємо параметри моделі
        theta -= learning_rate * gradient

    return theta

# Згенеруємо випадкові дані для регресії
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Додаємо стовпець одиниць для зсуву (bias term)
X_b = np.c_[np.ones((100, 1)), X]

# Викликаємо функцію для SGD
theta = linear_regression_sgd(X_b, y)
print("Значення параметрів моделі:", theta)