import numpy as np
from numpy.linalg import norm
from functools import wraps
from time import time

def power_vec(vector):
    return np.square(np.abs(vector))

def power_tot(vector):
    powers = power_vec(vector)
    return np.sum(powers)

def normalize_vec(vector):
    P = power_tot(vector)
    return vector / np.sqrt(P)

def normalize_pow(vector):
    return vector / np.sum(vector)

def MSE(vector1, vector2):
    vector1 = np.reshape(vector1, (-1,))
    vector2 = np.reshape(vector2, (-1,))    
    return np.sum(np.square(vector1 - vector2))/vector1.size