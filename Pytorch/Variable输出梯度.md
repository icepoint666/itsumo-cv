# Variable输出梯度
**By default, gradients are only retained for leaf variables, non-leaf variables' gradients are not retained to be inspected later. This was done by design, to save money.**

对于Variable，print命令是不能输出计算图非叶子节点的grad，为了去节省内存。
## 1.register_hook(print)方法
It can be used in 'register_hook(print)' method to print the gradients of intermediate variable but not 'print' method.

```python
import torch
from torch.autograd import Variable

x = Variable(torch.randn(2, 2), requires_grad = True)
y = 3 * x
z = y.sum(dim = 0)
loss = z.sum(dim = 0)
x.register_hook(print)
y.register_hook(print)
z.register_hook(print)
loss.register_hook(print)
loss.backward()

# Out:
# tensor(1.)
# tensor([1., 1.])
# tensor([[1., 1.],
#         [1., 1.]])
# tensor([[3., 3.],
#         [3., 3.]])
```
## 2.register_hook(print)方法

You can think of a function that also keeps some additional variables from the outer space.

For example in here 'hook' is a closure that remembers a name given to the outer function.

```python
import torch
from torch.autograd import Variable

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

x = Variable(torch.randn(2, 2), requires_grad = True)
y = 3 * x
z = y.sum(dim = 0)
loss = z.sum(dim = 0)
x.register_hook(save_grad('x'))
y.register_hook(save_grad('y'))
z.register_hook(save_grad('z'))
loss.register_hook(save_grad('loss'))
loss.backward()

print(grads)

# Out:{'loss': tensor(1.), 'z': tensor([1., 1.]), 'y': tensor([[1., 1.],
#         [1., 1.]]), 'x': tensor([[3., 3.],
#         [3., 3.]])}
```


https://discuss.pytorch.org
