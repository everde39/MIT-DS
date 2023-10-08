# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 13:25:18 2023

@author: Productive E-man
"""
import numpy as np
import project1 as p1
#%%Lecture 5 Linear Regression
'''
#Loss

x = np.array([[1,0,1], [1,1,1], [1,1,-1], [-1, 1, 1]])
y = np.array([2, 2.7, -0.7, 2])
theta = np.array([0, 1, 2])
n = 4

#margin = label*(np.dot(theta, feature_vector) + theta_0)
margin = y*(np.dot(x, theta))

#loss_h = max(0, 1-margin)
loss = np.maximum(0, 1-margin)

#avg_loss_h = np.mean(loss_h)
avg_loss = np.mean(loss)

print(avg_loss)

p1.hinge_loss_full(x, y, theta, 0)'''

#%%

x_1 = np.array([1, 0, 1]).T
x_2 = np.array([1, 1, 1]).T
x_3 = np.array([1, 1, -1]).T
x_4 = np.array([-1, 1, 1]).T

# Create an array of vectors
x = np.array([x_1, x_2, x_3, x_4]).T
theta = np.array([0,1,2]).T
y = np.array([2, 2.7, -0.7, 2])

#dp = np.dot(theta, x_4)


#%%


def hinge_loss(z):
    if z >= 1:
        return 0
    else:
        return 1 - z

def empirical_risk(theta, x, y):
    n = len(y)
    risk = 0
    
    for t in range(n):
        # Compute the dot product of theta and x[t]
        z = np.dot(theta, x[:, t])
        loss = hinge_loss(y[t] - z)
        risk += loss
    
    return risk / n


#%%
risk = empirical_risk(theta, x, y)
