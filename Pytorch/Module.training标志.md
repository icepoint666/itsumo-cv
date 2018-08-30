# Module.training 标志 如何影响 前向过程
从nn.Dropout 来看 Module.training
```python
class Dropout(Module):
    def __init__(self, p=0.5, inplace=False):
        super(Dropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(input, self.p, self.training, self.inplace)
```

可以看出，在forward 过程中，直接获取，父类的training的值。
我们 通常通过 module.train() 和 module.eval() 来切换模型的 训练测试阶段。
```python
def train(self, mode=True):
    """Sets the module in training mode.
    This has any effect only on modules such as Dropout or BatchNorm.
    """
    self.training = mode

    for module in self.children():
        # 递归调用子模块 train 函数， 来设定所有 module 的 training 值。
        module.train(mode)
        return self
```
需要注意的是：module.eval() 仅仅设置 module 的 training 属性，如果我们想获得最快的推断速度， 还需要 设置 输入 Variable的volatile 属性为 True。 

https://blog.csdn.net/u012436149/article/details/78281553
