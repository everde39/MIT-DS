
import numpy as np

theta = np.array([0, 0])
#x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
x = np.array([[1,0], [-1,1.5], [-1,-1]])

#y = np.array([[1], [-1], [1]])
y = np.array([[-1], [1], [1]])
list_theta = np.array([[0,0]])


for t in range(1001):
    for i in range(3):
        if y[i] * np.inner(x[i], theta) <= 0:
            theta = theta + y[i] * x[i]
            list_theta = np.append(list_theta, theta)


print(list_theta)
