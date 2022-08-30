import numpy as np

def gaussian(x, a, mean, sigma):
    return a * np.exp(-((x - mean) ** 2 / (2 * sigma**2))) / (sigma * np.sqrt(2 * np.pi))