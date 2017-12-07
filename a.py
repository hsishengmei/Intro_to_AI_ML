import numpy as np
a = [[1,2,3], [4,5,6], [7,8,9]]
a = np.array(a)
print a[:, 0]
print a[:, [0,2]]
a = np.delete(a, -1, 1)
print a