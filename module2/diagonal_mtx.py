import numpy as np

diagonal_elements = np.array([2, 3,4])
diagonal_matrix = np.diag(diagonal_elements)

print(diagonal_matrix)

inverse_matrix = np.linalg.inv(diagonal_matrix)

# Обчислення оберненої матриці за властивістю діагональної матриці
inverse_diagonal_matrix = np.diag( 1 / diagonal_elements)

print("\nОбернена матриця:")
print(inverse_matrix)

print("\nОбернена діагональна матриця:")
print(inverse_diagonal_matrix)