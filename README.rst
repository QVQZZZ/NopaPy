Download
--------

::

    pip install nopapy

!!ATTENTION!!
-------------
Check the `Project Homepage <https://github.com/QVQZZZ/NopaPy>`_ for the complete README.



Quick Start
-----------

::

    import numpy as np
    import nopapy as npp

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    ypred = npp.NWEstimate(x, y, 3.5) # supposed to be 7
    print(ypred) # 6.910633194984344

Other Modules
-------------

-  Kernel: Supports multiple built-in kernel functions and allows customization of kernel functions.
-  Estimate: Includes various non-parametric estimation methods, allowing customization of kernel functions, smooth bandwidth, and order.
-  Regression: Provides multiple smoothing methods for batch prediction.
-  Scikit\_like: Supports the generation of regression objects with specific parameters like scikit-learn and PyTorch.