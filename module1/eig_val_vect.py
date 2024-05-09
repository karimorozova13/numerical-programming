import numpy as np
from numpy import linalg as LA

input = np.array([[2,2],[8,2]])

# Власні значення, Власні вектори eigen
eig_val, eig_vect = LA.eig(input)
print(eig_val)
print(eig_vect)

first_vect = eig_vect[:,0]
print("first_vect: %s" %(first_vect))
second_vect = eig_vect[:,1]
print("second_vect: %s" %(second_vect))

# right vector - vector-column
# left vector - vector-row

