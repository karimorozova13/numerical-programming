import numpy as np

a = [2, -2, -3]
b = [4, 0, 6]
c = [-7, -7, 1]

# Мішаний добуток a, b і c
omega = np.linalg.det(np.dstack([a,b,c])) #volume of piramid V
print("omega: %s" %(omega))

