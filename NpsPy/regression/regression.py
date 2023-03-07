import numpy as np

from NpsPy.estimate.estimate import NWEstimate, GMEstimate, LPEstimate
from NpsPy.utils.check import check_input, check_regression
from NpsPy.utils.setting import DEFAULT_BANDWIDTH, DEFAULT_KERNEL, DEFAULT_DEGREE

__all__ = ["NWRegression", "GMRegression", "LPRegression"]

@check_input
@check_regression
def NWRegression(x: np.array, y: np.array, xs0: np.array, h=DEFAULT_BANDWIDTH, k=DEFAULT_KERNEL) -> np.array:
    """
    Nadaraya-Watson regression.
    Estimate the target values at xs0 using the kernel function k and bandwidth h.

    Parameters:
        x (np.array): Independent variable.
        y (np.array): Dependent variable.
        xs0 (np.array): Points at which to estimate the target values.
        h (float, optional): Bandwidth parameter. Defaults to DEFAULT_BANDWIDTH.
        k (callable, optional): Kernel function. Defaults to DEFAULT_KERNEL.

    Returns:
        np.array: Estimated values at given point xs0.
    """
    return np.array([NWEstimate(x, y, x0, h, k) for x0 in np.atleast_1d(xs0)])


@check_input
@check_regression
def GMRegression(x: np.array, y: np.array, xs0: np.array, h=DEFAULT_BANDWIDTH, k=DEFAULT_KERNEL) -> np.array:
    """
    Gasser-Muller regression.
    Estimate the target values at xs0 using the kernel function k and bandwidth h.

    Parameters:
        x (np.array): Independent variable.
        y (np.array): Dependent variable.
        xs0 (np.array): Points at which to estimate the target values.
        h (float, optional): Bandwidth parameter. Defaults to DEFAULT_BANDWIDTH.
        k (callable, optional): Kernel function. Defaults to DEFAULT_KERNEL.

    Returns:
        np.array: Estimated values at given point xs0.
    """
    return np.array([GMEstimate(x, y, x0, h, k) for x0 in np.atleast_1d(xs0)])


@check_input
@check_regression
def LPRegression(x: np.array, y: np.array, xs0: np.array, h=DEFAULT_BANDWIDTH, k=DEFAULT_KERNEL,
                 p=DEFAULT_DEGREE) -> np.array:
    """
    Local polynomial regression.
    Estimate the target values at xs0 using the kernel function k, bandwidth h and degree p.

    Args:
        x (np.array): Independent variable.
        y (np.array): Dependent variable.
        xs0 (np.array): Points at which to estimate the target values.
        h (float, optional): Bandwidth parameter. Defaults to DEFAULT_BANDWIDTH.
        k (callable, optional): Kernel function. Defaults to DEFAULT_KERNEL.
        p (int, optional): The degree of the local polynomial regression. Defaults to DEFAULT_DEGREE.

    Returns:
        np.array: Estimated values at given point xs0.

    Raises:
        ValueError: If the degree of local polynomial regression is negative or not an integer.
    """
    if p < 0 or not isinstance(p, int):
        raise ValueError("The degree of local polynomial regression p must be a non-negative integer.")
    return np.array([LPEstimate(x, y, x0, h, k, p) for x0 in np.atleast_1d(xs0)])