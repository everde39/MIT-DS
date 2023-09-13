#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 18:59:44 2023

@author: e-man
"""

import matplotlib.pyplot as plt # import the library
import numpy as np
import scipy as sp

#%% example
ex_x = np.array([1,2,3,4,5])
ex_y = np.array([10,20,30,40,50])

#%% Hubble Problem

Xs = np.array([0.0339, 0.0423, 0.213, 0.257, 0.273, 0.273, 0.450, 0.503, 0.503, \
0.637, 0.805, 0.904, 0.904, 0.910, 0.910, 1.02, 1.11, 1.11, 1.41, \
1.72, 2.03, 2.02, 2.02, 2.02])

Ys = np.array([-19.3, 30.4, 38.7, 5.52, -33.1, -77.3, 398.0, 406.0, 436.0, 320.0, 373.0, \
93.9, 210.0, 423.0, 594.0, 829.0, 718.0, 561.0, 608.0, 1.04E3, 1.10E3, \
840.0, 801.0, 519.0])

N = 24

#%%
mean_Xs = np.mean(Xs)
mean_Ys = np.mean(Ys)

print("mean_X", mean_Xs)
print("mean_y", mean_Ys)

#%%
ssdv_Xs = np.std(Xs, ddof=1)
ssdv_Ys = np.std(Ys, ddof=1)

print("sample_ssdX", ssdv_Xs)
print("sample_ssdY", ssdv_Ys)

#%%
cov_hubble =  np.cov(Xs, Ys, bias=False)[0,1]
print("cov_hubble", cov_hubble)

#%%
corr_hubble = np.corrcoef(Xs, Ys)[0, 1]
print("corr_hubble", corr_hubble)

#%%
import numpy as np

# Generate a random sample of x values from a standard normal distribution
np.random.seed(0)  # For reproducibility
x = np.random.normal(0, 1, 1000)  # Generating 1000 random samples from N(0,1)

# Calculate y values using the quadratic function y = x^2
y = x**2

# Calculate the correlation coefficient between x and y
correlation_coefficient = np.corrcoef(x, y)[0, 1]

# Print the correlation coefficient
print("Correlation Coefficient between x and y:", correlation_coefficient)

#%%
numerator_hubble = np.sum((Xs - mean_Xs) * (Ys - mean_Ys))
denominator_hubble = np.sum((Xs - mean_Xs) ** 2)

beta1_hat = numerator_hubble/denominator_hubble
print("Beta1_hat", beta1_hat)

#%%
beta0_hat = mean_Ys - (beta1_hat * mean_Xs)
print("Beta0_hat", beta0_hat)
