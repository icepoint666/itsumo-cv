# 复现styleGAN中pytorch用法积累

### 1. retain graph
反向传播的时候有时候设置retain_graph=True的作用是：
```python
    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
```

设置了retain_graph=True，这个参数的作用是什么，官方定义为：
```
    retain_graph (bool, optional) – If False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.
```
大意是如果设置为False，计算图中的中间变量在计算完后就会被释放。但是在平时的使用中这个参数默认都为False从而提高效率.

### 2.关于pytorch使用显存设备时变量的相关问题，以及多卡并行时数据

在pytorch运行是往往先定义一下设备
```python
_cuda = torch.cuda.is_available()
if _cuda:
    _device = torch.device("cuda", int(gpus[0]))
    _logger.info("Operation will be on *****GPU-CUDA***** ")
    print_cuda_statistics()
else:
    _device = torch.device("cpu")
    _logger.info("Operation will be on *****CPU***** ")
```
**问题1:** 这里的torch.device("cuda")指定的设备默认为"cuda: 0"

如果你想指定别的设备，或者指定cuda:2, cuda:3中的第一个显卡

那么可以这样操作：torch.device("cuda", 2)

在指定设备之后，需要把模型以及数据运过去，一般的操作
```python
# move to GPUs
G = G.to(_device)
D = D.to(_device)

real = real.to(_device)
```
**问题2：** 这里需要注意的是，运送模型G的时候，只是把里面的parameter运送过去，类型就是`torch.nn.Parameter`

如果你在里面定义的有torch.Tensor类型的变量，没有指定运送设备，所以就会导致，还是在cpu上的变量，不同设备上的变量之间使不能相互计算的

例如"cuda:2"与"cuda:3"的数据就不能进行计算，所以一般情况下是不要在模型内部的self里面定义Tensor

**问题3：** torch.cuda.Tensor类型与to("cuda")都只是把数据指定或者运送到一个固定的设备上

如果你定义的Tensor类型是`torch.cuda.Tensor`，那么它并不是每一个设备都有的，只是存在于默认的一个cuda设备

to("cuda")也是一样的道理

**问题4：** `nn.DataParallel（G， device_ids=[2, 3])`本质只是运送数据

对于模型如果实现并行，这个操作本身只是运送数据。

在定义阶段，这个指令，将模型参数运行到各个设备上。

在G模型forward运算的时候，这个模型会把输入数据并行copy到各个设备上，并行运算。

**问题5：** 数据并行的本质原理就是split，但是并不是按batch维度split

Tensor size如果是[8, 512, 64, 64]，那么如果是有两个卡，数据就会被split成为两个[4, 512, 64, 64]，分到两个卡上

问题就是如果数据size是[1, 512, 64, 64]，那么并不会复制两个然后送到各个设备上，而是split成为两个[1, 256, 64, 64]到数据上，这一点很容易出错

**问题6：** 不合理的并行计算，导致显存累积

显存累积就是随着训练的进行，你用nvidia-smi查看显存发现，每运行几秒，显存就上升一些，这就表示一些数据被无限copy到显卡中删除不掉

例如：像styleGAN那样，你希望在训练过程计算average dlatent variable，所以你就在训练循环的开始之前，定义了一个average dlatent变量，并且把它运到gpu上

```python
# average dlatent
dlatent_avg = torch.zeros([len(gpus), dlatent_size]).to(_device) # 这里你还提前写好了split的GPU数目，，来应对前面的split问题
```
然后接下来你在G 的forward的里面写入了这个变量作为输入，同时返回的值也有这个变量更新后的值

但是问题是，每次返回值后这个得到的更新后的average latents会在第二个显卡没法被销毁,第一个显卡因为事先有定义average dlatent，所以会去取代原来的位置。

这就会导致最终的现象就是：**第一个卡（也就是average latents定义在的那个卡）显存还是依旧维持稳定，随着训练的进行，但是第二个卡的显存会一直增加，直到爆显存。

**问题6 解决** 目前还没有寻找到解决办法

### 3. pytorch的dataloader本身就是一个python迭代器，所以不能无限循环
python迭代器的next操作，只能把这个循环迭代完就停止了，stopiteration

如果想让dataloader对数据无限迭代,在每次fetch数据的时候，代码改成这样
```python
# fetch forever
try:
    reals = next(data_iters[sched.resolution])
except:
    data_iters[sched.resolution] = iter((data_loaders[sched.resolution].train_loader))
    reals = next(data_iters[sched.resolution])
```

### 4. pytorch conv2d与transposeConv2d的一些本质

看源代码本质都是先定义一些torch.nn.Parameter类型的weight或者bias，然后通过F.conv2d以及F.transposeconv2d的操作

但是定义weight的时候只有transposeConv2d比较特别，weight维度是先input_channel,然后output_channel，然后kernel_size， kernel_size

对于conv2d以及其他Linear的weight定义都是先定义output_channel再定义input_channel

### 5. pytorch hub

PyTorch Hub是一个简易API和工作流程，为复现研究提供了基本构建模块，包含预训练模型库。

并且，PyTorch Hub还支持Colab，能与论文代码结合网站Papers With Code集成，用于更广泛的研究。

一行代码就导入

PyTorch Hub的使用简单到不能再简单，不需要下载模型，只用了一个torch.hub.load()就完成了对图像分类模型AlexNet的调用。
```python
import torch
model = torch.hub.load('pytorch/vision', 'alexnet', pretrained=True)
model.eval()
```
PyTorch Hub允许用户对已发布的模型执行以下操作：
```
    1、查询可用的模型;
    2、加载模型;
    3、查询模型中可用的方法。
```
下面让我们来看看每个应用的实例。
1、查询可用的模型

用户可以使用torch.hub.list()这个API列出repo中所有可用的入口点。比如你想知道PyTorch Hub中有哪些可用的计算机视觉模型：
```
>>> torch.hub.list('pytorch/vision')
>>>
['alexnet',
'deeplabv3_resnet101',
'densenet121',
...
'vgg16',
'vgg16_bn',
'vgg19',
 'vgg19_bn']
```
2、加载模型

在上一步中能看到所有可用的计算机视觉模型，如果想调用其中的一个，也不必安装，只需一句话就能加载模型。
```
model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
```
至于如何获得此模型的详细帮助信息，可以使用下面的API：
```
print(torch.hub.help('pytorch/vision', 'deeplabv3_resnet101'))
```
如果模型的发布者后续加入错误修复和性能改进，用户也可以非常简单地获取更新，确保自己用到的是最新版本：
```
model = torch.hub.load(..., force_reload=True)
```
对于另外一部分用户来说，稳定性更加重要，他们有时候需要调用特定分支的代码。例如pytorch_GAN_zoo的hub分支：
```
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'DCGAN', pretrained=True, useGPU=False)
```
3、查看模型可用方法


从PyTorch Hub加载模型后，你可以用dir(model)查看模型的所有可用方法。以bertForMaskedLM模型为例：
```
>>> dir(model)
>>>
['forward'
...
'to'
'state_dict',
]
```
如果你对forward方法感兴趣，使用help(model.forward) 了解运行运行该方法所需的参数。
```
>>> help(model.forward)
>>>
Help on method forward in module pytorch_pretrained_bert.modeling:
forward(input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None)
```
