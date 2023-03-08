import numpy as np
from scipy.integrate import quad

__all__ = ["I", "is_kernel"]

def I(x):
    return (np.abs(x) <= 1).astype(int)

def is_kernel(func, tol=1e-8):
    x = np.linspace(-5, 5, num=1000)
    y = func(x)
    if not np.all(y >= 0):
        return False
    if not np.allclose(y, y[::-1], atol=tol):
        return False
    if not np.isclose(quad(func, -np.inf, np.inf)[0], 1, atol=tol):
        return False
    return True