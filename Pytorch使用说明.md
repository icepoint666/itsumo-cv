# Pytorch
## Overview
### 明确定义了三层抽象：

![](../__pics/pytorch.png)

### Pytorch: Tensor
定义方式，前向传播，反向传播，更新权值的方式

![](../__pics/pytorch1.png)

使用gpu计算时的实现方式：

![](../__pics/pytorch2.png)

### Pytorch: Tensor

![](../__pics/pytorch3.png)

这里x是一个变量，那么x.data就是一个tensor

x.grad就是另一个变量，包含了损失

x.grad.data就是包含梯度的Tensor

在Pytorch上的Tensor和Variable有相同的API

![](../__pics/pytorch4.png)

定义的时候告诉构造器，该不该在变量上计算梯度：

![](../__pics/pytorch5.png)

##### Tensorflow与Pytorch
 - tensorflow是先构建显示的图，之后计算
 
 - Pytorch是每次前向传播时都要构建一个新图，这个程序看上去要简单一点 

### 自定义函数
在Pytorch我们可以自己定义变量的前向和后向，
 
可以构造新的autograd函数

![](../__pics/pytorch6.png)

![](../__pics/pytorch7.png)

但是，大多数的时候，你不需要自己定义函数，因为一些高层wrappers已经把这些实现都准备好了

## Pytorch: nn
### 很类似keras的model

![](../__pics/pytorch8.png)

### 使用optimizer训练

![](../__pics/pytorch9.png)

通过optimizer.step()迭代更新

### Pytorch: Modules
pytorch一般定义模型都要新建一个类

![](../__pics/pytorch10.png)

### Pytorch: DataLoaders
DataLoaders可以处理建立mini batches，也可以处理多线程的问题

事实上它可以用多线程建立很多批处理

每次迭代都可以产生分批的数据，然后在其内部重排数据，多线程加载数据

![](../__pics/pytorch11.png)

### Pytorch: Pretrained Model
大概是最容易使用Pretrained Model的方法

![](../__pics/pytorch12.png)

### Pytorch: visdom
可视化很多损失统计，类似于Tensorboard

但是Tensorboard可以让你可视化很多计算图的过程，visdom还没有这个功能

![](../__pics/pytorch13.png)

### Tensorflow静态图，Pytorch动态图

![](../__pics/pytorch14.png)

Tensorflow(static graph)中事先构建图，之后每次跑都不用再构建图，所以当多次使用图的时候，效率会比较高

但是Dynamic graph灵活性更好，例如下面的例子：条件判定

![](../__pics/pytorch15.png)

如果遇到条件判定，像Tensorflow这样的静态图，会产生一个很多的分支，所有控制流路径都要提前建立好，
例如需要Tensorflow的流操作：eg: tf.cond
但是Dynamic就会像Python控制流一样判定

另一个例子：循环

![](../__pics/pytorch16.png)

这样的情况Pytorch只需要使用一个循环，并不依赖于我们输入数据的大小

Tensorflow实现就需要特殊的控制流：

![](../__pics/pytorch17.png)
From Stanford.cs231n
