import pprint
import scipy as sc

matrix_A = sc.array([[7,4],[3,5]])
P, L, U = sc.linalg.lu(matrix_A)

print("Original matrix A:")
pprint.pprint(matrix_A)

# implies a pivoting(reordering) rows(or columns) in case it is needed
print("Pivoting matrix P:")
pprint.pprint(P)

# lower-triangular matrix of A
print("L:")
pprint.pprint(L)

# upper-triangular matrix of A
print("U:")
pprint.pprint(U)