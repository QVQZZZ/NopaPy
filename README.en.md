## Download
`
pip install NpsPy
`

## Quick Start
```
import numpy as np
import npspy

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
ypred = npspy.NWEstimate(x, y, 3.5) # supposed to be 7
print(ypred) # 6.910633194984344
```