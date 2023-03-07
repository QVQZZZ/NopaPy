import functools
import warnings

import numpy as np

from npr.utils.setting import DEFAULT_BANDWIDTH, DEFAULT_KERNEL


def check_estimate(func):
    """ Decorator to check the x0 input of the function. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y, x0, *args_ = args
        if not isinstance(x0, (int, float, np.int32)):
            raise TypeError(f"{func.__name__}() expects x0 to be an integer or float or numpy.int32.")
        return func(*args, **kwargs)

    return wrapper


def check_regression(func):
    """ Decorator to check the xs0 input of the function. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y, xs0, *args_ = args
        if not isinstance(xs0, (np.ndarray, list)):
            corr_func = func.__name__.split("Regression")[0] + "Estimate()"
            warn_msg = f"{func.__name__}() expects xs0 to be np.ndarray or list. " \
                       f"Your code won't make any errors, " \
                       f"but we still recommend that you use {corr_func} instead. "
            warnings.warn(warn_msg)

        elif len(xs0) == 1:
            corr_func = func.__name__.split("Regression")[0] + "Estimate()"
            _ = xs0[0]
            warn_msg = f"{func.__name__}() expects xs0 to be a vector, not a scalar. " \
                       f"You may have passed an incorrect value for parameter xs0, " \
                       f"If you want to estimate the dependent variable of a point, " \
                       f"use {corr_func} instead of {func.__name__}, " \
                       f"and use 'x0={_}' instead of 'x0=[{_}]'. "
            warnings.warn(warn_msg)

        return func(*args, **kwargs)

    return wrapper


def check_input(func):
    """ Decorator to check the inputs """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        x, y, *args_ = args
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(f"{func.__name__}() expects x and y to be np.ndarray.")
        if not x.shape == y.shape:
            raise ValueError(f"{func.__name__}() expects x and y to have the same shape.")
        if not isinstance(kwargs.get('h', DEFAULT_BANDWIDTH), (float, int)):
            raise ValueError(f"{func.__name__}() expects h to be a float or int.")
        if kwargs.get('h', DEFAULT_BANDWIDTH) <= 0:
            raise ValueError(f"{func.__name__}() expects h to be a positive number.")
        if not callable(kwargs.get('k', DEFAULT_KERNEL)):
            raise TypeError(f"{func.__name__}() expects k to be a function.")
        return func(*args, **kwargs)

    return wrapper