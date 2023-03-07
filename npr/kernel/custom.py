import numpy as np
from scipy.integrate import quad

__all__ = ["I", "is_kernel"]

def I(x):
    return (np.abs(x) <= 1).astype(int)

def is_kernel(func, tol=1e-8):
    x = np.linspace(-5, 5, num=1000)
    y = func(x)
    # 检查是否非负
    if not np.all(y >= 0):
        return False
    # 检查是否对称
    if not np.allclose(y, y[::-1], atol=tol):
        return False
    # 检查是否积分为 1
    if not np.isclose(quad(func, -np.inf, np.inf)[0], 1, atol=tol):
        return False
    return True