## 下载
`
pip install nonpapy
`

## 快速开始
```
import numpy as np
import nonpapy

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
ypred = nonpapy.NWEstimate(x, y, 3.5) # supposed to be 7
print(ypred) # 6.910633194984344
```

## 其他功能模块
- kernel: 支持多种内置核函数，同时支持自定义核函数
- estimate: 包含多种非参数估计方法，允许自定义核函数、光滑带宽、阶数
- regression: 提供多种光滑方法进行批量预测
- scikit_like: 支持像scikit-learn和PyTorch生成特定参数的回归对象，传递自变量即可获得预测值