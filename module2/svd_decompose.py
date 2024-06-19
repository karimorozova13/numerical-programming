import numpy as np
import matplotlib.pyplot as plt

matrix_a = np.array([[4, 0],
                    [3, -5]])

matrix_a_transposed = np.transpose(matrix_a)

res = np.dot(matrix_a_transposed, matrix_a)

print("Матриця А: \n", matrix_a)
print("Матриця А транспонована: \n", matrix_a_transposed)
print("Добуток матриці A на її транспоновану: \n", res)

values, vecs = np.linalg.eig(res)

print("Вектори: \n", vecs)
print("Значення: \n", values)

sorted_indices  = np.argsort(vecs[:, 0])
V = vecs[sorted_indices]
print("Вектори, відсортовані за першим елементом: \n", V)

Sigma = np.diag(values)
Sigma = np.sqrt(Sigma)
print("Діагональна матриця: \n", Sigma)

Sigma_transpose = Sigma.transpose()
print("Транспонована діагональна матриця: \n", Sigma_transpose)

m = np.array([[1,2],
          [3,4]]).T
print("Матриця А: \n", m)

AV = np.dot(matrix_a, V)
print("Матриця А * V: \n", AV)

def matrix_morm(mtr):
    sqrt_sum_columns = np.sqrt(np.sum(mtr**2, axis=0))
    res = mtr / sqrt_sum_columns
    return res

AV = matrix_morm(AV)
print("Матриця А * V: \n", AV)

U = np.dot(AV, Sigma_transpose)
U = matrix_morm(U)
print("Матриця U: \n", U)

A = np.dot(np.dot(U, Sigma), V.T)
print("Матриця А: \n", A)

print(U)
print(Sigma)
print(V)


Uu, Ss, Vh = np.linalg.svd(matrix_a)
print('SVD U ')
print(Uu)
print('SVD Sigma ')
print(Ss)
print('SVD Vh ')
print(Vh)

k = 2
plt.plot(np.arange(k), Ss[:k])
plt.xlabel('Rank of singular value')
plt.ylabel('Magnitude of singular value')
plt.show()