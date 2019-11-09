import numpy as np
import math

def returnFunc(a):
    return a

a = np.array([1,2,3,4,5])
c = np.array([[1,2,3,4,5], [1,2,3,4,5], [1,2,3,4,5]])
b = 3
d = 3
e = np.array([[2,3], [2,3],[2,1], [2,1]])

print(np.multiply(-b,a))
print(np.exp(10))
print(a.dot(np.matrix.transpose(c)))
#print(np.matrix.transpose(a).dot(c))
print(a[0])
print(b/d)
print(e[:,1:])
print(returnFunc(a))
n_samples, n_features = e.shape
print("sample: ", n_samples)
print("features: ", n_features)

f = []
for i in range(n_samples):
    x1 = e[i][0]
    x2 = e[i][1]
    f.append([1, x1, x2, math.pow(x1, 2), x1 * x2, math.pow(x2, 2), math.pow(x1, 3), math.pow(x1, 2) * x2,x1 * math.pow(x2, 2), math.pow(x2, 3)])

print(f)

