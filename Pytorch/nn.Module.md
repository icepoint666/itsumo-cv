# nn.Module
#### nn.Module基类的构造函数：
```python
def __init__(self):
    self._parameters = OrderedDict()
    self._modules = OrderedDict()
    self._buffers = OrderedDict()
    self._backward_hooks = OrderedDict()
    self._forward_hooks = OrderedDict()
    self.training = True
```
其中每个属性的解释如下：
_parameters：字典，保存用户直接设置的parameter，self.param1 = nn.Parameter(t.randn(3, 3))会被检测到，在字典中加入一个key为'param'，value为对应parameter的item。而self.submodule = nn.Linear(3, 4)中的parameter则不会存于此。
_modules：子module，通过self.submodel = nn.Linear(3, 4)指定的子module会保存于此。
_buffers：缓存。如batchnorm使用momentum机制，每次前向传播需用到上一次前向传播的结果。
_backward_hooks与_forward_hooks：钩子技术，用来提取中间变量，类似variable的hook。
training：BatchNorm与Dropout层在训练阶段和测试阶段中采取的策略不同，通过判断training值来决定前向传播策略。
上述几个属性中，_parameters、_modules和_buffers这三个字典中的键值，都可以通过self.key方式获得，效果等价于self._parameters['key'].
 
定义一个Module，这个Module即包含自己的Parameters有包含子Module及其Parameters，
```python
import torch as t
from torch import nn
from torch.autograd import Variable as V
 
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 等价与self.register_parameter('param1' ,nn.Parameter(t.randn(3, 3)))
        self.param1 = nn.Parameter(t.rand(3, 3))
        self.submodel1 = nn.Linear(3, 4) 
    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel11(x)
        return x
net = Net()
```
### 1. _modules
打印网络对象的话会输出子module结构
```python
print(net)
Net(
  (submodel1): Linear(in_features=3, out_features=4)
)
# ._modules输出的也是子module结构，不过数据结构和上面的有所不同
print(net.submodel1)
print(net._modules) # 字典子类
Linear(in_features=3, out_features=4)
OrderedDict([('submodel1', Linear(in_features=3, out_features=4))])
for name, submodel in net.named_modules():
    print(name, submodel)
 Net(
  (submodel1): Linear(in_features=3, out_features=4)
)
submodel1 Linear(in_features=3, out_features=4)
print(list(net.named_modules())) # named_modules其实是包含了本层的module集合
[('', Net(
  (submodel1): Linear(in_features=3, out_features=4)
)), ('submodel1', Linear(in_features=3, out_features=4))]
```
### 2. _parameters
```python
# ._parameters存储的也是这个结构
print(net.param1)
print(net._parameters) # 字典子类，仅仅包含直接定义的nn.Parameters参数
Parameter containing:
 0.6135  0.8082  0.4519
 0.9052  0.5929  0.2810
 0.6825  0.4437  0.3874
[torch.FloatTensor of size 3x3]

OrderedDict([('param1', Parameter containing:
 0.6135  0.8082  0.4519
 0.9052  0.5929  0.2810
 0.6825  0.4437  0.3874
[torch.FloatTensor of size 3x3]
)])
 
for name, param in net.named_parameters():
    print(name, param.size())
param1 torch.Size([3, 3])
submodel1.weight torch.Size([4, 3])
submodel1.bias torch.Size([4])
```

### 3. _buffers
```python
bn = nn.BatchNorm1d(2)
input = V(t.rand(3, 2), requires_grad=True)
output = bn(input)
bn._buffers
OrderedDict([('running_mean', 
              1.00000e-02 *
                9.1559
                1.9914
              [torch.FloatTensor of size 2]), ('running_var', 
               0.9003
               0.9019
              [torch.FloatTensor of size 2])])
```
### 4. training
```python
input = V(t.arange(0, 12).view(3, 4))
model = nn.Dropout()
# 在训练阶段，会有一半左右的数被随机置为0
model(input)
Variable containing:
  0   2   4   0
  8  10   0   0
  0  18   0  22
[torch.FloatTensor of size 3x4]
```
```python
model.training  = False
# 在测试阶段，dropout什么都不做
model(input)
Variable containing:
  0   1   2   3
  4   5   6   7
  8   9  10  11
[torch.FloatTensor of size 3x4]
```
Module.train()、Module.eval() 方法和 Module.training属性的关系
```python
print(net.training, net.submodel1.training)
net.train() # 将本层及子层的training设定为True
net.eval() # 将本层及子层的training设定为False
net.training = True # 注意，对module的设置仅仅影响本层，子module不受影响
net.training, net.submodel1.training
True True
(True, False)
```
