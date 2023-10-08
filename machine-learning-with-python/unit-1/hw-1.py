#Perceptron Script
import numpy as np

theta = np.array([0, 0])
list_theta = np.array([[0,0]])

#x = np.array([[-1, -1], [1, 0], [-1, 10]])     #array of x in order x1-x3
#x = np.array([[1,0], [-1,10], [-1,-1]])         #array of x in order x2-x1
x = np.array([[-4,2], [-2,1], [-1,-1], [2,2], [1,-2]])

#y = np.array([[1], [-1], [1]])
#y = np.array([[-1], [1], [1]])
y = np.array([[1], [1], [-1], [-1], [-1]])

#times_misclassified = np.array([[1], [0], [2], [1], [0]])

#for t in range(1001):
for i in range(1):
    if y[i] * np.inner(x[i], theta) <= 0:
        theta = theta + y[i] * x[i]
        list_theta = np.append(list_theta, theta)


print(list_theta)



# [[-1,-1],[-2,9],[-3,8],[-4,7],[-5,6],[-6,5]]


#%%
import numpy as np

def train_perceptron(x, y, theta_init, theta_0_init, max_iterations=1000):
    theta = theta_init.copy()
    theta_0 = theta_0_init
    list_theta = [np.hstack([theta_0, theta])]  # Store the initial theta and theta_0
    num_samples = x.shape[0]

    for t in range(max_iterations):
        misclassified = False
        for i in range(num_samples):
            if y[i] * (np.dot(x[i], theta) + theta_0) <= 0:
                theta += y[i] * x[i]
                theta_0 += y[i]
                list_theta.append(np.hstack([theta_0, theta]))
                misclassified = True

        if not misclassified:
            break

    return list_theta

# Example input data and labels without bias term in x
x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
y = np.array([1, 1, -1, -1, -1])

# Initial values for theta (excluding bias term) and bias term (theta_0)
theta_init = np.array([0.0, 0.0])
theta_0_init = 0.0

# Train the perceptron
list_theta = train_perceptron(x, y, theta_init, theta_0_init)

# Print the list of theta values during training
for i, theta in enumerate(list_theta):
    print(f"Iteration {i}: theta_0={theta[0]}, theta={theta[1:]}")

#%%
import numpy as np

# Given data
d = 2
y = np.array([1, 1])
theta = np.zeros(d)
x1 = np.cos(np.pi * 1)  # x^(1) for t=1
x2 = np.cos(np.pi * 2)  # x^(2) for t=2

# Initialize variables
num_updates = 0

# Perceptron algorithm
for t in range(1, d + 1):
    if y[t - 1] * np.dot(theta, np.array([x1, x2])) <= 0:
        theta += y[t - 1] * np.array([x1, x2])
        num_updates += 1

    if t == 1:  # Reset theta after the first update
        theta = np.zeros(d)

# Print the number of updates and theta components
print("Number of updates:", num_updates)
print("theta_1:", theta[0])
print("theta_2:", theta[1])


