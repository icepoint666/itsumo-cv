# pytorch0.4.0与1.0.1版本切换的问题

**torch.utils.ffi**

被直接弃用，里面的build_extension等一些方法，都被1.0.1中的下述方法替代
```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
```
对于原来里面的_wrap_function函数，没有替代

**所以，如果想将自己定义的c++/cuda模块，安装在python环境里**

**cuda代码里的头文件**

replace `#include <torch/torch.h>` by `#include <torch/extension.h>`

假如自定义module源代码是torch0.4.0，你的环境是torch1.0.1

参照flownet2的两个库里面的自定义模块的对比

`NVIDIA/flownet2_pytorch`与`vt-vl-lab/pytorch-flownet2`
