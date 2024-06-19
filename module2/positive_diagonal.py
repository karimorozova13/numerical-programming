import numpy as np

matrix = np.array([[4, -1, 2],
                   [0, 5, 3],
                   [0, 0, 6]])

if all(matrix[i, i] > 0 for i in range(matrix.shape[0])):
    print("Матриця є позитивно оптимальною.")
else:
    print("Матриця не є позитивно оптимальною.")