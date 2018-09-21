## 把Pytorch当做Numpy来用

PyTorch的官方介绍是一个拥有强力GPU加速的张量和动态构建网络的库，其主要构建是张量，所以可以把PyTorch当做Numpy来用，Pytorch的很多操作好比Numpy都是类似的，但是其能够在GPU上运行，所以有着比Numpy快很多倍的速度。
```python
import torch
import numpy as np

# 创建一个numpy ndarray
numpy_tensor = np.random.randn(10, 20)

# 我们可以使用两种方式将numpy的ndarray转换到tensor上
pytorch_tensor1 = torch.Tensor(numpy_tensor)
pytorch_tensor2 = torch.from_numpy(numpy_tensor)
```

#### 将pytorch tensor转换为numpy ndarray
pytorch tensor在cpu上
```python
numpy_array = pytorch_tensor1.numpy()
```
GPU上的Tensor不能直接转换为Numpy ndarray，需要使用.cpu()先将GPU上的Tensor转到CPU上

#### PyTorch Tensor 使用GPU加速 
可以使用下面两种方法将Tensor放到GPU上
```python
# 第一种方式是定义cuda数据类型
dtype = torch.cuda.FloatTensor
gpu_tensor = torch.randn(10,20).type(dtype)

# 第二种方式更简单，推荐使用
#gpu_tensor = torch.randn(10,20).cuda(0) # 将tensor放到第一个GPU上
#gpu_tensor = torch.randn(10,20).cuda(1) # 将tensor放到第二个GPU上
```
使用第一种方式将tensor放到GPU上的时候会将数据类型转换成定义的类型，而是用第二种方式能够直接将tensor放到GPU上，类型跟之前保持一致
推荐在定义tensor的时候就明确数据类型，然后直接使用第二种方法将tensor放到GPU上

例如GTX 960M，可能不支持第二种方式。

#### demo
```python
# 使用的是GPU版本
x = torch.randn(3,2).type(torch.cuda.DoubleTensor)
x_array = x.cpu().numpy()
print(x_array.dtype)


float64

```

https://blog.csdn.net/qjk19940101/article/details/79555653

