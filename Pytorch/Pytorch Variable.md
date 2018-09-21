# Pytorch Variable
tensor是PyTorch中的完美组件，但是构建神经网络还远远不够，我们需要能够构建计算图的tensor，这就是Variable。

Variable是对tensor的封装，操作和tensor是一样的，但是每个Variable都有三个属性，Variable中的tensor本身.data，对应tensor的梯度.grad以及这个Variable是通过说明方式得到的.grad_fn
```python
# 通过下面这种方式导入Variable
from torch.autograd import Variable

x_tensor = torch.randn(10, 5)
y_tensor = torch.randn(10, 5)

# 将tensor变成Variable 
x = Variable(x_tensor, requires_grad=True) # 默认Variable 是不需要求梯度的，所以用这个方式申明需要对其进行求梯度
y = Variable(y_tensor, requires_grad=True)

z = torch.sum(x+y)

print(z.data)
print(z.grad_fn)
```
```python
 14.4502
[torch.FloatTensor of size 1]

<SumBackward0 object at 0x7fb3787ed518>
```

上面我们打出了z中的tensor数值，同时通过grad_fn知道了其是通过Sum这种方式得到的
```python
# 求x和y的梯度
z.backward()
print(x.grad)
print(y.grad)
```

```

Variable containing:
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
[torch.FloatTensor of size 10x5]

Variable containing:
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
    1     1     1     1     1
[torch.FloatTensor of size 10x5]
```

通过.grad我们得到了x和y的梯度，这里我们使用了Pytorch提供的自动求导机制，非常的方便，下一小节会自动将自动求导


https://blog.csdn.net/qjk19940101/article/details/79555653
