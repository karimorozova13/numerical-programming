import numpy as np

orthogonal_matrix = np.array([[1/3, -2/3, 2/3],
                               [2/3, 2/3, 1/3],
                               [-2/3, 1/3, 2/3]])

inverse_orthogonal_matrix = np.linalg.inv(orthogonal_matrix)

print("Ортогональна матриця:")
print(orthogonal_matrix)

print("\nОбернена матриця:")
print(inverse_orthogonal_matrix)

determinant = np.linalg.det(orthogonal_matrix)

print(determinant)