import numpy as np

matrix = np.array([[2, -1, 0],
                   [-1, 2, -1],
                   [0, -1, 2]])

eig_vals, _ = np.linalg.eig(matrix)

if all(eig_val > 0 for eig_val in eig_vals):
    print("Матриця є позитивно оптимальною.", [eig_val for eig_val in eig_vals])
else:
    print("Матриця не є позитивно оптимальною.")

# Розмір матриці
n = matrix.shape[0]

# Перевірка знаку лівих верхніх піддетермінантів
positive_definite = all(np.linalg.det(matrix[:i, :i]) for i in range(1, n + 1))

if positive_definite:
    print("Матриця є додатно визначеною.", [np.linalg.det(matrix[:i, :i]) for i in range(1, n + 1)])
else:
    print("Матриця не є додатно визначеною.")