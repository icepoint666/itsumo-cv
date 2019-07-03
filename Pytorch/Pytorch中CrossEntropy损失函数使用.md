# Pytorch中CrossEntropy损失函数使用

### nn.CrossEntropy
针对单目标分类问题, 结合了 `nn.LogSoftmax()` 和 `nn.NLLLoss()` 来计算 loss.

用于训练类别classes 的分类问题.

参数`weight`是 `1D Tensor`, 分别对应每个类别class的权重. 对于类别不平衡的训练数据集比较有用.

输入`input` 包含了每一类别的概率或score.

输入 input Tensor 的大小是 `(minibatch,C)` 或 `(minibatch,C,d1,d2,...,dK)`. `K≥2` 表示 K-dim 场景

输入 target 是类别 class 的索引([0,C−1], C 是类别classes 总数.)

**注意这里与Tensorflow的cross_entropy loss存在区别，tensorflow的input与target的tensor维度是一样的，然而pytorch是cross_entropy的loss只针对单独的类别。**

### F.binary_cross_entropy_with_logits
可以作为cross entropy loss的替代使用，输入与输出维度要求是一样的，可以用于多目标类的分类


```
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=None, reduction='mean', pos_weight=None):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool], Optional[bool], str, Optional[Tensor]) -> Tensor
    r"""Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'mean'
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
```
