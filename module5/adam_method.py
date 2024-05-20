# Adaptive Moment Estimation

import numpy as np
np.random.seed(42)


def adam_optimizer(X, y, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10000):
    m = X.shape[0]  # кількість прикладів
    n = X.shape[1]  # кількість ознак

    # Ініціалізація параметрів моделі
    theta = np.random.randn(n, 1)

    # Ініціалізація моментів
    m_t = np.zeros((n, 1))
    v_t = np.zeros((n, 1))

    for iteration in range(num_iterations):
        # Обчислення градієнту
        gradients = 2/m * X.T.dot(X.dot(theta) - y)

        # Оновлення моментів
        m_t = beta1 * m_t + (1 - beta1) * gradients
        v_t = beta2 * v_t + (1 - beta2) * (gradients ** 2)

        # Коригування зміщення моментів
        m_t_hat = m_t / (1 - beta1 ** (iteration + 1))
        v_t_hat = v_t / (1 - beta2 ** (iteration + 1))

        # Оновлення параметрів моделі
        theta = theta - learning_rate * m_t_hat / (np.sqrt(v_t_hat) + epsilon)

    return theta

# Приклад використання методу Adam для оптимізації лінійної регресії
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Додаємо стовбець одиниць до матриці X
X_b = np.c_[np.ones((100, 1)), X]

# Застосовуємо оптимізатор Adam для навчання моделі
theta_optimized = adam_optimizer(X_b, y)

print("Оптимальні параметри моделі за допомогою методу Adam:")
print(theta_optimized)
