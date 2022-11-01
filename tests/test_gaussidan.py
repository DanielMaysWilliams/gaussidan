import numpy as np

from gaussidan import __version__, fit_gaussian, plot_gaussian


def test_version():
    assert __version__ == "0.1.1"


def test_weights():
    s = np.random.normal(0, 1, 10000)
    n, bins = np.histogram(s, bins=10)
    height, mu, sigma = fit_gaussian(s, bins)
    bins_x = (bins[:-1] + bins[1:]) / 2
    height_w, mu_w, sigma_w = fit_gaussian(bins_x, bins, n)

    assert np.isclose(height, height_w)
    assert np.isclose(mu, mu_w)
    assert np.isclose(sigma, sigma_w)


def test_normal_fit():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000000)
    height, mu, sigma = fit_gaussian(data, bins=100)

    assert np.isclose(height, 95106.04488674947)
    assert np.isclose(mu, -0.0015697600096978438)
    assert np.isclose(sigma, 1.0012873211511928)


def test_normal_plot():
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000000)
    x, y = plot_gaussian(data, bins=10)

    assert np.allclose(
        x,
        np.array(
            [
                -4.35401675,
                -3.40317824,
                -2.45233973,
                -1.50150122,
                -0.55066271,
                0.4001758,
                1.35101431,
                2.30185282,
                3.25269133,
                4.20352984,
            ]
        ),
    )
    assert np.allclose(
        y,
        np.array(
            [
                5.56384293e01,
                1.70170415e03,
                2.24979942e04,
                1.28574276e05,
                3.17625176e05,
                3.39176921e05,
                1.56562668e05,
                3.12392557e04,
                2.69441003e03,
                1.00456314e02,
            ]
        ),
    )


def test_binomial_fit():
    np.random.seed(42)
    data = np.random.binomial(2**15, 0.5, 1000000)
    height, mu, sigma = fit_gaussian(data, bins=100)

    assert np.isclose(height, 9161706.365818698)
    assert np.isclose(mu, 16384.04506823164)
    assert np.isclose(sigma, 90.57466541369749)


def test_binomial_plot():
    np.random.seed(42)
    data = np.random.binomial(2**15, 0.5, 1000000)
    x, y = plot_gaussian(data, bins=10)

    assert np.allclose(
        x,
        np.array(
            [
                15968.8,
                16060.4,
                16152.0,
                16243.6,
                16335.2,
                16426.8,
                16518.4,
                16610.0,
                16701.6,
                16793.2,
            ]
        ),
    )
    assert np.allclose(
        y,
        np.array(
            [
                2.36684633e01,
                1.06322316e03,
                1.86160343e04,
                1.27045209e05,
                3.37938593e05,
                3.50368996e05,
                1.41586526e05,
                2.23011141e04,
                1.36911388e03,
                3.27612877e01,
            ]
        ),
    )
