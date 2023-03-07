import numpy as np

from NpsPy.kernel.custom import I

__all__ = ["GaussianKernel", "EpanechnikovKernel", "tricubeKernel", "boxcarKernel"]

GAUSSIAN_COEFFICIENT = 1 / np.sqrt(2 * np.pi)
EPANECHNIKOV_COEFFICIENT = 0.75
TRICUBE_COEFFICIENT = 70 / 81
BOX_COEFFICIENT = 0.5

def GaussianKernel(x):
    return GAUSSIAN_COEFFICIENT * np.exp(-np.power(x, 2) / 2)


def EpanechnikovKernel(x):
    return EPANECHNIKOV_COEFFICIENT * (1 - np.power(x, 2)) * I(x)


def tricubeKernel(x):
    return TRICUBE_COEFFICIENT * np.power(1 - np.power(np.abs(x), 3), 3) * I(x)


def boxcarKernel(x):
    return BOX_COEFFICIENT * I(x)