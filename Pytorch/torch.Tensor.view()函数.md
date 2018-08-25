# tensor.view()变换得到的 tensor 一致
features为2D-Tensor,即需要将 3D-Tensor的RGB 图片或者是2D-Tensor的灰度图片拉伸成1D-Tensor,在使用的时候再还原。一缩一放之后的 tensor是否一致呢？
```python
import torch

a = torch.rand(10,3,4,5)
b = a.view(10,-1)
c = b.view(10,3,4,5)

print(torch.equal(c, a))
# True
```

输出结果为 True,前后缩放的 Tensor一致。

https://blog.csdn.net/u011394059/article/details/78207717?locationNum=4&fps=1
