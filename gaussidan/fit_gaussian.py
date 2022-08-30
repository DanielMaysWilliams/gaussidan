from typing import Sequence, Union

import numpy as np
from scipy.optimize import curve_fit

from gaussidan.utils import gaussian

def fit_gaussian(
    data: Union[Sequence[float], np.ndarray],
    bins: Union[int, Union[Sequence[float], np.ndarray]],
    weights: Union[Sequence[float], np.ndarray] = None,
) -> tuple[float, float, float]:

    hist, bin_edges = np.histogram(data, bins=bins, weights=weights)

    mask = np.nonzero(hist)
    hist = hist[mask]
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers = bin_centers[mask]

    popt, _ = curve_fit(gaussian, bin_centers, hist)
    height, mu, sigma = popt

    return height, mu, sigma
