# 遇到问题“ImportError: torch.utils.ffi is deprecated. Please use cpp extensions instead.”的解决方案

原因： 在PyTorch 1.0.1中，torch.utils.ffi被弃用了，需要用其他包来替代。

解决办法： 对于博主遇到的问题，将原语句
```
from torch.utils.ffi import create_extension
```
修改成
```
from torch.utils.cpp_extension import BuildExtension
```
再将文件下面的调用
```
ffi = create_extension(...)
```
改成
```
ffi = BuildExtension(...)
```
原文链接：https://blog.csdn.net/ShuqiaoS/article/details/88420326
