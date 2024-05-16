import numpy as np


# Вектори
a = np.array([2, -2, -3])
b = np.array([4, 0, 6])
c = np.array([-7, -7, 1])

# Мішаний добуток a, b і c
mixed_dot_product = np.linalg.det(np.dstack([a,b,c]))
print(mixed_dot_product)


# Векторний добуток b і c
cross_product = np.cross(b, c)

# Мішаний добуток a, b і c
mixed_dot_product = np.dot(a, cross_product)

# Виведення результату
print("Мішаний добуток:", mixed_dot_product)
