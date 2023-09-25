
import numpy as np

theta = np.array([0, 0])
#x = np.array([[-1, -1], [1, 0], [-1, 10]])     #array of x in order x1-x3
x = np.array([[1,0], [-1,10], [-1,-1]])         #array of x in order x2-x1

#y = np.array([[1], [-1], [1]])
y = np.array([[-1], [1], [1]])
list_theta = np.array([[0,0]])


for t in range(1001):
    for i in range(3):
        if y[i] * np.inner(x[i], theta) <= 0:
            theta = theta + y[i] * x[i]
            list_theta = np.append(list_theta, theta)


print(list_theta)

# [[-1,-1],[-2,9],[-3,8],[-4,7],[-5,6],[-6,5]]