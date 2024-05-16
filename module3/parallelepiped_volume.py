import numpy as np

def parallelepiped_volume(a, b, c):
    # Обчислення мішаного добутку
    mixed_dot_product = np.dot(a, np.cross(b, c))

    # Обчислення об'єму паралелепіпеда
    volume = abs(mixed_dot_product)

    return volume

# Приклад використання
a = np.array([2, -2, -3])
b = np.array([4, 0, 6])
c = np.array([-7, -7, 1])

volume = parallelepiped_volume(a, b, c)
print("Об'єм паралелепіпеда:", volume)
