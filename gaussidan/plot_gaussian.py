from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from gaussidan import fit_gaussian
from gaussidan.utils import gaussian

def plot_gaussian(
    data: Union[Sequence[float], np.ndarray],
    bins: Union[int, Union[Sequence[float], np.ndarray]],
    weights: Union[Sequence[float], np.ndarray] = None,
):

    height, mu, sigma = fit_gaussian(data, bins, weights)
    _, bin_edges, _ = plt.hist(data, bins=bins, weights=weights)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.plot(bin_centers, gaussian(bin_centers, height, mu, sigma))
    plt.show()
