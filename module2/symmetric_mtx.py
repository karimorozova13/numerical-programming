import numpy as np


# Симетрична матриця    
A = np.array([[4, 1, 2],
              [1, 5, 3],
              [2, 3, 6]])

eig_vals, eig_vects = np.linalg.eig(A)

# Формування діагональної матриці з власних значень
D= np.diag(eig_vals)

# Формування матриці власних векторів
V= eig_vects

# Відновлення початкової матриці за допомогою розкладу
reconstructed = np.dot(V, np.dot(D, V.T))

print("Початкова симетрична матриця A:")
print(A)
print("\nДіагональна матриця власних значень D:")
print(D)
print("\nМатриця власних векторів V:")
print(V)
print("\nВідновлена матриця з розкладу:")
print(reconstructed)

print("\nПеревірка, що матриця власних векторів є ортогональною:")

# np.allclose - перевірка рівності аргументів із заданою точністю
orth = np.allclose(np.dot(V, V.T), np.eye(3))
# orth = np.allclose(np.dot(eig_vects.T, eig_vects), np.eye(A.shape[0]))

print(np.dot(eig_vects.T, eig_vects))

print(np.eye(A.shape[0]))

if orth:
    print("Власні вектори є ортогональними.")
else:
    print("Власні вектори не є ортогональними.")