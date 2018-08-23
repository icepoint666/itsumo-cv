# lr_scheduler
## lr_scheduler
lr_scheduler是Pytorch2.0新发布的功能，相比于broadcast等更新的众望所归，我认为这一小更新还是非常实用的。以前训练的时候只能固定一个学习率或者自己写代码实现，lr_scheduler的出现可谓填补了Pytorch框架的最后一块砖。
## 基类_LRScheduler
总体上说，_LRScheduler是所有学习率改变策略的基类，输入为optimizer对象，当训练不是从第一代开始（中断后继续训练），还需要输入last_epoch（默认为-1）。与Optimizer类似的是，其主要功能体现在step()方法中，用于更新optimizer对象每个param_group字典的lr键的值。 
下面先从基类的代码开始理解：
```python
class _LRScheduler(object):
def __init__(self, optimizer, last_epoch=-1):
    # 输入的对象必须为Optimizer
    if not isinstance(optimizer, Optimizer):
        raise TypeError('{} is not an Optimizer'.format(
            type(optimizer).__name__))
    self.optimizer = optimizer
    # 如果是第一代，则在param group中新建一个键：'initial_lr'
    if last_epoch == -1:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    else:
        # 如果不是第一代，则必须有'initial_lr'这个键
        for i, group in enumerate(optimizer.param_groups):
            if 'initial_lr' not in group:
                raise KeyError("param 'initial_lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
    # base_lrs是每个parm_group字典中'initial_lr'的值
    self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    self.step(last_epoch + 1)
    self.last_epoch = last_epoch

# 用于计算每个parm_group在当前epoch的学习率
def get_lr(self):
    raise NotImplementedError

def step(self, epoch=None):
    # step可以记录epoch的值，也可以外部输入
    if epoch is None:
        epoch = self.last_epoch + 1
    self.last_epoch = epoch
    # 更新每个param_group字典中键lr的值。每个键lr的值可以不同，由get_lr()来实现
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
        param_group['lr'] = lr  
```
## LambdaLR
LambdaLR用于设计自己所需的学习率更新策略。一个官方的例子如下：
```python
#Assuming optimizer has two groups.
lambda1 = lambda epoch: epoch // 30
lambda2 = lambda epoch: 0.95 ** epoch
scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
for epoch in range(100):
     scheduler.step()
     train(...)
     validate(...)  
LambdaLR需要指定lr_lambda，其合法的值为单一的lambda表达式，或者由与parm_group字典数目相同的lambda表达式构成的list。
class LambdaLR(_LRScheduler):
def __init__(self, optimizer, lr_lambda, last_epoch=-1):
    self.optimizer = optimizer
    if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
        # 如果lr_lambda为单一的lambda表达式，则将其复制扩充
        self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
    else:
        # 确保每个param_group有与之对应的lambda表达式（一一对应）
        if len(lr_lambda) != len(optimizer.param_groups):
            raise ValueError("Expected {} lr_lambdas, but got {}".format(
                len(optimizer.param_groups), len(lr_lambda)))
        self.lr_lambdas = list(lr_lambda)
    self.last_epoch = last_epoch
    super(LambdaLR, self).__init__(optimizer, last_epoch)

# 计算每个param_group在当前epoch下的lr
def get_lr(self):
    return [base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)] 
```
## StepLR
StepLR即固定步长学习率指数衰减策略。其实现的效果如下：
```python
# Assuming optimizer uses lr = 0.5 for all groups，
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
# ...
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    scheduler.step()
    train(...)
    validate(...)  
```
StepLR代码实现较为简单，需要指定的是步长step_size与衰减率gamma：
```python
class StepLR(_LRScheduler):
def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
    self.step_size = step_size
    self.gamma = gamma
    super(StepLR, self).__init__(optimizer, last_epoch)

def get_lr(self):
    # //表示取整除运算
    return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs] 
```
其他策略与上面2种代表性的策略大同小异，不作赘述。
一个比较有意思的策略是ReduceLROnPlateau，它能自动调节学习率，不需要事先给定策略。其大致思路是：“Reduce learning rate when a metric has stopped improving”。所以需要在调用step()时传入一个反应学习效果的指标metrics。代码较为复杂且不常用，故只给出一个官方的例子：
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
for epoch in range(10):
    train(...)
    val_loss = validate(...)
    # Note that step should be called after validate()
    scheduler.step(val_loss)
```

http://blog.leanote.com/post/timeandpressure/Pytorch.lr_scheduler-2
