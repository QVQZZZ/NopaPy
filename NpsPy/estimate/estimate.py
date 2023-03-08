import numpy as np
from scipy import integrate

from npspy.utils.check import check_input, check_estimate
from npspy.utils.setting import DEFAULT_BANDWIDTH, DEFAULT_KERNEL, DEFAULT_DEGREE

__all__ = ["NWEstimate", "GMEstimate", "LPEstimate"]

@check_input
@check_estimate
def NWEstimate(x: np.array, y: np.array, x0, h=DEFAULT_BANDWIDTH, k=DEFAULT_KERNEL) -> float:
    """
    Nadaraya-Watson estimate.
    Estimate the target value at x0 using the kernel function k and bandwidth h.

    Parameters:
        x (np.array): Independent variable.
        y (np.array): Dependent variable.
        x0 (float): Point at which to estimate the target value.
        h (float, optional): Bandwidth parameter. Defaults to DEFAULT_BANDWIDTH.
        k (callable, optional): Kernel function. Defaults to DEFAULT_KERNEL.

    Returns:
        float: Estimated value at given point x0.
    """
    l = k((x0 - x) / h) / sum(k((x0 - x) / h))
    return sum(l * y)

@check_input
@check_estimate
def GMEstimate(x: np.array, y: np.array, x0, h=DEFAULT_BANDWIDTH, k=DEFAULT_KERNEL) -> float:
    """
    Gasser-Muller estimate.
    Estimate the target value at x0 using the kernel function k and bandwidth h.

    Parameters:
        x (np.array): Independent variable.
        y (np.array): Dependent variable.
        x0 (float): Point at which to estimate the target value.
        h (float, optional): Bandwidth parameter. Defaults to DEFAULT_BANDWIDTH.
        k (callable, optional): Kernel function. Defaults to DEFAULT_KERNEL.

    Returns:
        float: Estimated value at given point x0.
    """
    s = [-np.inf] + [0.5 * (x[i] + x[i + 1]) for i in range(len(x) - 1)] + [np.inf]
    return sum([integrate.quad(k, (s[i] - x0) / h, (s[i + 1] - x0) / h)[0] * y[i] for i in range(len(x))])




@check_input
@check_estimate
def LPEstimate(x: np.array, y: np.array, x0, h=DEFAULT_BANDWIDTH, k=DEFAULT_KERNEL, p=DEFAULT_DEGREE) -> float:
    """
    Local polynomial estimate.
    Estimate the target values at x0 using the kernel function k, bandwidth h and degree p.

    Args:
        x (np.array): Independent variable.
        y (np.array): Dependent variable.
        x0 (float): Point at which to estimate the target value.
        h (float, optional): Bandwidth parameter. Defaults to DEFAULT_BANDWIDTH.
        k (callable, optional): Kernel function. Defaults to DEFAULT_KERNEL.
        p (int, optional): The degree of the local polynomial regression. Defaults to DEFAULT_DEGREE.

    Returns:
        float: Estimated value at given point x0.

    Raises:
        ValueError: If the degree of local polynomial regression is negative or not an integer.
    """
    if p < 0 or not isinstance(p, int):
        raise ValueError("The degree of local polynomial regression p must be a non-negative integer.")
    X = np.array([[((x[i] - x0) ** j) for j in range(p + 1)] for i in range(len(x))])
    W = np.diag(k((x - x0) / h))
    l = np.dot(np.insert(np.zeros(p), 0, 1), np.linalg.pinv(X.T.dot(W).dot(X)).dot(X.T).dot(W))
    return sum(l * y)