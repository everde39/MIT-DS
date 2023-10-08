# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 12:39:55 2023

@author: Productive E-man
"""
import numpy as np
import project1 as p1

#%%

# Define example values for the arguments
feature_vector = np.array([2.0, 3.0])  # Feature vector with two features
label = 1  # True label (+1 or -1 for binary classification)
theta = np.array([0.5, -0.5])  # Weight vector
theta_0 = -1.0  # Offset term

# Call the hinge_loss_single function with the example values
loss_h = p1.hinge_loss_single(feature_vector, label, theta, theta_0)

# Print the result
print("Hinge Loss:", loss_h)

#%%

import numpy as np

# Define example values for the arguments
feature_matrix = np.array([[2.0, 3.0], [1.0, -1.0], [-1.0, 2.0]])  # Feature matrix with three data points
labels = np.array([1, -1, 1])  # True labels for the three data points
theta = np.array([0.5, -0.5])  # Weight vector
theta_0 = -1.0  # Offset term

# Call the hinge_loss_full function with the example values
loss_h = p1.hinge_loss_full(feature_matrix, labels, theta, theta_0)

# Print the hinge losses for each data point
for i, loss in enumerate(loss_h):
    print(f"Hinge Loss for Data Point {i + 1}: {loss}")
