from gaussidan import __version__
from gaussidan import fit_gaussian

import numpy as np


def test_version():
    assert __version__ == '0.1.0'

def test_weights():
    s = np.random.normal(0, 1, 10000)
    n, bins = np.histogram(s, bins=10)
    height, mu, sigma = fit_gaussian(s, bins)
    bins_x = (bins[:-1] + bins[1:])/2
    height_w, mu_w, sigma_w = fit_gaussian(bins_x, bins, n)
    
    assert np.isclose(height, height_w)
    assert np.isclose(mu, mu_w)
    assert np.isclose(sigma, sigma_w)
