import numpy as np

def gradient(func, x):
    x = np.array(x, dtype=float)
    
    # Ініціалізуємо градієнт з усіма нулями такоголементами
    grad = np.zeros_like(x)
    # Обчислюємо часткові похідні для кожної змінної
    # Використовуємо центральну різницеву схему для чисельного обчислення похідних
    h = 1e-6  # дуже мале число для обчислення похідних

    for i in range(len(x)):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        grad[i] = (func(*x_plus_h) - func(*x)) / h

    return grad

# Приклад використання:

def my_function(x, y, z):
    return 3*(x**2 + y*z)

point = (2, 3, 4)
grad = gradient(my_function, point)
print("Градієнт у точці", point, ":", grad)