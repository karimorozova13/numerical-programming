import numpy as np

def normal_vector(u, v):
    # Обчислення векторного добутку (cross product)
    n = np.cross(u, v)

    return n

# Приклад використання
u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

normal = normal_vector(u, v)
print("Вектор нормалі до площини:", normal)
