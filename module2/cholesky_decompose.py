import numpy as np
from numpy.linalg import cholesky

A = np.array([[36, 30, 18], [30, 41, 23], [18, 23, 14]])
print(A)

L = cholesky(A)
print(L)
print(L.T)

B = L.dot(L.T)
print(B)