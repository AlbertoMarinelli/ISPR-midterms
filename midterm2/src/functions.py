import numpy as np

def sigmoid(x):
    ones = [1.] * len(x)
    return np.divide(ones, np.add (ones, np.exp(-x)))