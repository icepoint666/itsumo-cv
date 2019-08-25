# pytorch冻结想冻结的预训练模型中的某一层

原链接：

https://www.zhihu.com/question/311095447/answer/589307812

https://zhuanlan.zhihu.com/p/76459295

**优先方法**

在加载预训练模型的时候，我们有时想冻结前面几层，使其参数在训练过程中不发生变化。

我们需要先知道每一层的名字，通过如下代码打印：
```python
net = Network()  # 获取自定义网络结构
for name, value in net.named_parameters():
    print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
```

假设前几层信息如下：
```
name: cnn.VGG_16.convolution1_1.weight,	 grad: True
name: cnn.VGG_16.convolution1_1.bias,	 grad: True
name: cnn.VGG_16.convolution1_2.weight,	 grad: True
name: cnn.VGG_16.convolution1_2.bias,	 grad: True
name: cnn.VGG_16.convolution2_1.weight,	 grad: True
name: cnn.VGG_16.convolution2_1.bias,	 grad: True
name: cnn.VGG_16.convolution2_2.weight,	 grad: True
name: cnn.VGG_16.convolution2_2.bias,	 grad: True
```

后面的True表示该层的参数可训练，然后我们定义一个要冻结的层的列表：
```
no_grad = [
    'cnn.VGG_16.convolution1_1.weight',
    'cnn.VGG_16.convolution1_1.bias',
    'cnn.VGG_16.convolution1_2.weight',
    'cnn.VGG_16.convolution1_2.bias'
]
```

冻结方法如下：
```python
net = Net.CTPN()  # 获取网络结构
for name, value in net.named_parameters():
    if name in no_grad:
        value.requires_grad = False
    else:
        value.requires_grad = True
```

冻结后我们再打印每层的信息：
```
name: cnn.VGG_16.convolution1_1.weight,	 grad: False
name: cnn.VGG_16.convolution1_1.bias,	 grad: False
name: cnn.VGG_16.convolution1_2.weight,	 grad: False
name: cnn.VGG_16.convolution1_2.bias,	 grad: False
name: cnn.VGG_16.convolution2_1.weight,	 grad: True
name: cnn.VGG_16.convolution2_1.bias,	 grad: True
name: cnn.VGG_16.convolution2_2.weight,	 grad: True
name: cnn.VGG_16.convolution2_2.bias,	 grad: True
```

可以看到前两层的weight和bias的requires_grad都为False，表示它们不可训练。


最后在定义优化器时，只对requires_grad为True的层的参数进行更新。
```python
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)
```

**下面有4种可行参考方法，最后一种方法的完整代码实现**

首先假设如下的模型：
```python
class Char3SeqModel(nn.Module):
    
    def __init__(self, char_sz, n_fac, n_h):
        super().__init__()
        self.em = nn.Embedding(char_sz, n_fac)
        self.fc1 = nn.Linear(n_fac, n_h)
        self.fc2 = nn.Linear(n_h, n_h)
        self.fc3 = nn.Linear(n_h, char_sz)
        
    def forward(self, ch1, ch2, ch3):
        # do something
        out = #....
        return out

model = Char3SeqModel(10000, 50, 25)
```

假设需要冻结fc1，有如下几个方法 

方法1：
```python
# 冻结
model.fc1.weight.requires_grad = False
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.1)
# 
# compute loss 
loss.backward()
optmizer.step()

# 解冻
model.fc1.weight.requires_grad = True
optimizer.add_param_group({'params': model.fc1.parameters()})
```
方法2：
```python
# 冻结
optimizer = optim.Adam([{'params':[ param for name, param in model.named_parameters() if 'fc1' not in name]}], lr=0.1)
# compute loss
loss.backward()
optimizer.step()

# 解冻
optimizer.add_param_group({'params': model.fc1.parameters()})
```

方法3：

大体思路：将原来的layer的weight缓存下来，每次反向传播之后，再将原来的weight赋值给相应的layer。
```python
fc1_old_weights = Variable(model.fc1.weight.data.clone())
# compute loss
loss.backward()
optimizer.step()
model.fc1.weight.data = fc1_old_weights.data
```

方法4：

大体思路：在每次进行反向传播更新权重之前将相应layer的gradient手动置为0。缺点也很明显，会浪费计算资源。
```python
# compute loss
loss.backward()
# set fc1 gradients to 0
optimizer.step()
```

**终极方法代码实现：**
```python
from collections.abc import Iterable

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, idxs, freeze=True):
    if not isinstance(idxs, Iterable):
        idxs = [idxs]
    num_child = len(list(model.children()))
    idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, idxs))
    for idx, child in enumerate(model.children()):
        if idx not in idxs:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze
            
def freeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, True)

def unfreeze_by_idxs(model, idxs):
    set_freeze_by_idxs(model, idxs, False)
```
```python
# 冻结第一层
freeze_by_idxs(model, 0)
# 冻结第一、二层
freeze_by_idxs(model, [0, 1])
#冻结倒数第一层
freeze_by_idxs(model, -1)
# 解冻第一层
unfreeze_by_idxs(model, 0)
# 解冻倒数第一层
unfreeze_by_idxs(model, -1)
```
```python
# 冻结 em层
freeze_by_names(model, 'em')
# 冻结 fc1, fc3层
freeze_by_names(model, ('fc1', 'fc3'))
# 解冻em, fc1, fc3层
unfreeze_by_names(model, ('em', 'fc1', 'fc3'))
```
