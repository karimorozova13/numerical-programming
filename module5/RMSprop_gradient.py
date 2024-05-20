# Root Mean Square Propagation

import numpy as np
np.random.seed(42)

# Генеруємо випадкові дані для демонстрації
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Додаємо стовбець одиниць до матриці X
X_b = np.c_[np.ones((100, 1)), X]

# Випадково ініціалізуємо вектор параметрів theta
theta = np.random.randn(2,1)

# Встановлюємо гіперпараметри методу RMSprop
eta = 0.1  # крок навчання
gamma = 0.9  # коефіцієнт зглажування
epsilon = 1e-8  # малий додаток для чисельної стійкості
grad_squared = 0  # зберігаємо квадрати градієнтів

# Реалізація методу RMSprop
for iteration in range(1000):
    gradients = 2/100 * X_b.T.dot(X_b.dot(theta) - y)
    grad_squared = gamma * grad_squared + (1 - gamma) * gradients**2
    theta = theta - eta * gradients / (np.sqrt(grad_squared) + epsilon)

# Виведення оцінених параметрів моделі
print("theta:")
print(theta)
