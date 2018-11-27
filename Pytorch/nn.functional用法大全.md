# torch.nn.functional

## Convolution 函数
```
torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
对几个输入平面组成的输入信号应用1D卷积。

有关详细信息和输出形状，请参见Conv1d。

参数：
- input – 输入张量的形状 (minibatch x in_channels x iW)
- weight – 过滤器的形状 (out_channels, in_channels, kW)
- bias – 可选偏置的形状 (out_channels)
- stride – 卷积核的步长，默认为1

例子：
```python
    >>> filters = autograd.Variable(torch.randn(33, 16, 3))
    >>> inputs = autograd.Variable(torch.randn(20, 16, 50))
    >>> F.conv1d(inputs, filters)
```

```
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
对几个输入平面组成的输入信号应用2D卷积。

有关详细信息和输出形状，请参见Conv2d。

参数：
- input – 输入张量 (minibatch x in_channels x iH x iW)
- weight – 过滤器张量 (out_channels, in_channels/groups, kH, kW)
- bias – 可选偏置张量 (out_channels)
- stride – 卷积核的步长，可以是单个数字或一个元组 (sh x sw)。默认为1
- padding – 输入上隐含零填充。可以是单个数字或元组。 默认值：0
- groups – 将输入分成组，in_channels应该被组数除尽

例子：
```python
    >>> # With square kernels and equal stride
    >>> filters = autograd.Variable(torch.randn(8,4,3,3))
    >>> inputs = autograd.Variable(torch.randn(1,4,5,5))
    >>> F.conv2d(inputs, filters, padding=1)
```
```
torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```
对几个输入平面组成的输入信号应用3D卷积。

有关详细信息和输出形状，请参见Conv3d。

参数：
- input – 输入张量的形状 (minibatch x in_channels x iT x iH x iW)
- weight – 过滤器张量的形状 (out_channels, in_channels, kT, kH, kW)
- bias – 可选偏置张量的形状 (out_channels)
- stride – 卷积核的步长，可以是单个数字或一个元组 (sh x sw)。默认为1
- padding – 输入上隐含零填充。可以是单个数字或元组。 默认值：0

例子：
```python
    >>> filters = autograd.Variable(torch.randn(33, 16, 3, 3, 3))
    >>> inputs = autograd.Variable(torch.randn(20, 16, 50, 10, 20))
    >>> F.conv3d(inputs, filters)
```
```
torch.nn.functional.conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)
```
```
torch.nn.functional.conv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)
```
在由几个输入平面组成的输入图像上应用二维转置卷积，有时也称为“去卷积”。

有关详细信息和输出形状，请参阅ConvTranspose2d。

参数：
- input – 输入张量的形状 (minibatch x in_channels x iH x iW)
- weight – 过滤器的形状 (in_channels x out_channels x kH x kW)
- bias – 可选偏置的形状 (out_channels)
- stride – 卷积核的步长，可以是单个数字或一个元组 (sh x sw)。默认: 1
- padding – 输入上隐含零填充。可以是单个数字或元组。 (padh x padw)。默认: 0
- groups – 将输入分成组，in_channels应该被组数除尽
- output_padding – 0 <= padding <stride的零填充，应该添加到输出。可以是单个数字或元组。默认值：0
```
torch.nn.functional.conv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1)
```
在由几个输入平面组成的输入图像上应用三维转置卷积，有时也称为“去卷积”。

有关详细信息和输出形状，请参阅ConvTranspose3d。

参数：
- input – 输入张量的形状 (minibatch x in_channels x iT x iH x iW)
- weight – 过滤器的形状 (in_channels x out_channels x kH x kW)
- bias – 可选偏置的形状 (out_channels)
- stride – 卷积核的步长，可以是单个数字或一个元组 (sh x sw)。默认: 1
- padding – 输入上隐含零填充。可以是单个数字或元组。 (padh x padw)。默认: 0

## Pooling 函数
```
torch.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```
对由几个输入平面组成的输入信号进行一维平均池化。

有关详细信息和输出形状，请参阅AvgPool1d。

参数：
- kernel_size – 窗口的大小
- stride – 窗口的步长。默认值为kernel_size
- padding – 在两边添加隐式零填充
- ceil_mode – 当为True时，将使用ceil代替floor来计算输出形状
- count_include_pad – 当为True时，这将在平均计算时包括补零

例子：
```python
    >>> # pool of square window of size=3, stride=2
    >>> input = Variable(torch.Tensor([[[1,2,3,4,5,6,7]]]))
    >>> F.avg_pool1d(input, kernel_size=3, stride=2)
    Variable containing:
    (0 ,.,.) =
      2  4  6
    [torch.FloatTensor of size 1x1x3]
```
```
torch.nn.functional.avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
```
在kh x kw区域中应用步长为dh x dw的二维平均池化操作。输出特征的数量等于输入平面的数量。

有关详细信息和输出形状，请参阅AvgPool2d。

参数：
- input – 输入的张量 (minibatch x in_channels x iH x iW)
- kernel_size – 池化区域的大小，可以是单个数字或者元组 (kh x kw)
- stride – 池化操作的步长，可以是单个数字或者元组 (sh x sw)。默认等于核的大小
- padding – 在输入上隐式的零填充，可以是单个数字或者一个元组 (padh x padw)，默认: 0
- ceil_mode – 定义空间输出形状的操作
- count_include_pad – 除以原始非填充图像内的元素数量或kh * kw
```
torch.nn.functional.avg_pool3d(input, kernel_size, stride=None)
```
在kt x kh x kw区域中应用步长为dt x dh x dw的二维平均池化操作。输出特征的数量等于 input planes / dt。
```
torch.nn.functional.max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

torch.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

torch.nn.functional.max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)

torch.nn.functional.max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)

torch.nn.functional.max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)

torch.nn.functional.max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None)

torch.nn.functional.lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)
```
```
torch.nn.functional.adaptive_max_pool1d(input, output_size, return_indices=False)
```
在由几个输入平面组成的输入信号上应用1D自适应最大池化。

有关详细信息和输出形状，请参阅AdaptiveMaxPool1d。

参数：
- output_size – 目标输出大小（单个整数）
- return_indices – 是否返回池化的指数
```
torch.nn.functional.adaptive_max_pool2d(input, output_size, return_indices=False)
```
在由几个输入平面组成的输入信号上应用2D自适应最大池化。

有关详细信息和输出形状，请参阅AdaptiveMaxPool2d。

参数：
- output_size – 目标输出大小（单整数或双整数元组）
- return_indices – 是否返回池化的指数
```
torch.nn.functional.adaptive_avg_pool1d(input, output_size)
```
在由几个输入平面组成的输入信号上应用1D自适应平均池化。

有关详细信息和输出形状，请参阅AdaptiveAvgPool1d。

参数：
- output_size – 目标输出大小（单整数或双整数元组）
```
torch.nn.functional.adaptive_avg_pool2d(input, output_size)
```
在由几个输入平面组成的输入信号上应用2D自适应平均池化。

有关详细信息和输出形状，请参阅AdaptiveAvgPool2d。

参数：
- output_size – 目标输出大小（单整数或双整数元组）

## 非线性激活函数
```
torch.nn.functional.threshold(input, threshold, value, inplace=False)
```
```
torch.nn.functional.relu(input, inplace=False)
```
```
torch.nn.functional.hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)
```
```
torch.nn.functional.relu6(input, inplace=False)
```
```
torch.nn.functional.elu(input, alpha=1.0, inplace=False)
```
```
torch.nn.functional.leaky_relu(input, negative_slope=0.01, inplace=False)
```
```
torch.nn.functional.prelu(input, weight)
```
```
torch.nn.functional.rrelu(input, lower=0.125, upper=0.3333333333333333, training=False, inplace=False)
```
```
torch.nn.functional.logsigmoid(input)
```
```
torch.nn.functional.hardshrink(input, lambd=0.5)
```
```
torch.nn.functional.tanhshrink(input)
```
```
torch.nn.functional.softsign(input)
```
```
torch.nn.functional.softplus(input, beta=1, threshold=20)
```
```
torch.nn.functional.softmin(input)
```
```
torch.nn.functional.softmax(input)
```
```
torch.nn.functional.softshrink(input, lambd=0.5)
```
```
torch.nn.functional.log_softmax(input)
```
```
torch.nn.functional.tanh(input)
```
```
torch.nn.functional.sigmoid(input)
```

## Normalization 函数
```
torch.nn.functional.batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)
```
## 线性函数
```
torch.nn.functional.linear(input, weight, bias=None)
```
## Dropout 函数
```
torch.nn.functional.dropout(input, p=0.5, training=False, inplace=False)
```
## 距离函数（Distance functions）
```
torch.nn.functional.pairwise_distance(x1, x2, p=2, eps=1e-06)
```
计算向量v1、v2之间的距离（成次或者成对，意思是可以计算多个，可以参看后面的参数）

参数：

    x1:第一个输入的张量
    x2:第二个输入的张量
    p:矩阵范数的维度。默认值是2，即二范数。

规格：

    输入:(N,D)其中D等于向量的维度
    输出:(N,1)

例子：
```python
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> output = F.pairwise_distance(input1, input2, p=2)
    >>> output.backward()
```
## 损失函数（Loss functions）
```
torch.nn.functional.nll_loss(input, target, weight=None, size_average=True)
```
负的log likelihood损失函数. 详细请看NLLLoss.

参数：
- input - (N,C) C 是类别的个数
- target - (N) 其大小是 0 <= targets[i] <= C-1
- weight (Variable, optional) – 一个可手动指定每个类别的权重。如果给定的话，必须是大小为nclasses的Variable
- size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。

Variables:
- weight – 对于constructor而言，每一类的权重作为输入
```
torch.nn.functional.kl_div(input, target, size_average=True)
```
KL 散度损失函数，详细请看KLDivLoss

参数：
- input – 任意形状的 Variable
- target – 与输入相同形状的 Variable
- size_average – 如果为TRUE，loss则是平均值，需要除以输入 tensor 中 element 的数目
```
torch.nn.functional.cross_entropy(input, target, weight=None, size_average=True)
```
该函数使用了 log_softmax 和 nll_loss，详细请看CrossEntropyLoss

参数：
- input - (N,C) 其中，C 是类别的个数
- target - (N) 其大小是 0 <= targets[i] <= C-1
- weight (Variable, optional) – 一个可手动指定每个类别的权重。如果给定的话，必须是大小为nclasses的Variable
- size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。
```
torch.nn.functional.binary_cross_entropy(input, target, weight=None, size_average=True)
```
该函数计算了输出与target之间的二进制交叉熵，详细请看BCELoss

参数：
- input – 任意形状的 Variable
- target – 与输入相同形状的 Variable
- weight (Variable, optional) – 一个可手动指定每个类别的权重。如果给定的话，必须是大小为nclasses的Variable
- size_average (bool, optional) – 默认情况下，是mini-batchloss的平均值，然而，如果size_average=False，则是mini-batchloss的总和。
```
torch.nn.functional.smooth_l1_loss(input, target, size_average=True)
```
## Vision functions
```
torch.nn.functional.pixel_shuffle(input, upscale_factor)[source]
```
将形状为[*, C*r^2, H, W]的Tensor重新排列成形状为[C, H*r, W*r]的Tensor.

详细请看PixelShuffle.

形参说明:
- input (Variable) – 输入
- upscale_factor (int) – 增加空间分辨率的因子.

例子:
```python
    ps = nn.PixelShuffle(3)
    input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
    output = ps(input)
    print(output.size())
    torch.Size([1, 1, 12, 12])
```
```
torch.nn.functional.pad(input, pad, mode='constant', value=0)[source]
```
填充Tensor.

目前为止,只支持2D和3D填充.
Currently only 2D and 3D padding supported.
当输入为4D Tensor的时候,pad应该是一个4元素的tuple (pad_l, pad_r, pad_t, pad_b ) ,当输入为5D Tensor的时候,pad应该是一个6元素的tuple (pleft, pright, ptop, pbottom, pfront, pback).

形参说明:
- input (Variable) – 4D 或 5D tensor
- pad (tuple) – 4元素 或 6-元素 tuple
- mode – ‘constant’, ‘reflect’ or ‘replicate’
- value – 用于constant padding 的值.

https://blog.csdn.net/HowardWood/article/details/79508805
