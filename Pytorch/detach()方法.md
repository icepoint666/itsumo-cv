# detach方法.md
## detach

官方文档中，对这个方法是这么介绍的。

返回一个新的 从当前图中分离的 Variable。

返回的 Variable 永远不会需要梯度

如果 被 detach 的Variable volatile=True， 那么 detach 出来的 volatile 也为 True

还有一个注意事项，即：返回的 Variable 和 被 detach 的Variable 指向同一个 tensor

```python
import torch
from torch.nn import init
from torch.autograd import Variable
t1 = torch.FloatTensor([1., 2.])
v1 = Variable(t1)
t2 = torch.FloatTensor([2., 3.])
v2 = Variable(t2)
v3 = v1 + v2
v3_detached = v3.detach()
v3_detached.data.add_(t1) # 修改了 v3_detached Variable中 tensor 的值
print(v3, v3_detached)    # v3 中tensor 的值也会改变
```
```python
# detach 的源码
def detach(self):
    result = NoGrad()(self)  # this is needed, because it merges version counters
    result._grad_fn = None
    return result
```

## detach_

官网给的解释是：将 Variable 从创建它的 graph 中分离，把它作为叶子节点。

从源码中也可以看出这一点

将 Variable 的grad_fn 设置为 None，这样，BP 的时候，到这个 Variable 就找不到 它的 grad_fn，所以就不会再往后BP了。

将 requires_grad 设置为 False。这个感觉大可不必，但是既然源码中这么写了，如果有需要梯度的话可以再手动 将 requires_grad 设置为 true
```python
# detach_ 的源码
def detach_(self):
    """Detaches the Variable from the graph that created it, making it a
    leaf.
    """
    self._grad_fn = None
    self.requires_grad = False
```

https://blog.csdn.net/u012436149/article/details/76714349 
