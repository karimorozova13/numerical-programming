import numpy as np

vec_c = np.cross([4, -5, 0], [0,4, -3])
cp = np.linalg.norm(vec_c)
print("cp: %s" %(cp))
