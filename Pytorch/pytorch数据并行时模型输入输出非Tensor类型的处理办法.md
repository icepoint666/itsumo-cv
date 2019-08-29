# pytorch数据并行时模型输入输出非Tensor类型的处理办法

有时候深度学习模型输入或者输出的部分并不是tensor类型的数据，可能是一个list,tuple,dict，而且里面的每个元素的size大小可能都是不一致的。

这种情况下对于例如下面数据并行（Data parallel)的操作中,可能会出现问题

```python
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
```
首先了解一下**pytorch中data_parallel函数原理**

源代码：
```python
def data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None):
    r"""Evaluates module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if device_ids is None:
        device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    inputs, module_kwargs = scatter_kwargs(inputs, module_kwargs, device_ids, dim)
    if len(device_ids) == 1:
        return module(*inputs[0], **module_kwargs[0])
    used_device_ids = device_ids[:len(inputs)]
    replicas = replicate(module, used_device_ids)
    outputs = parallel_apply(replicas, inputs, module_kwargs, used_device_ids)
    return gather(outputs, output_device, dim)
```
总的来说一共分为三个步骤:
- 首先是scatter，将input的数据散播
- 然后是parallel_apply，将每个并行设备的input放入模型中参与运算
- 最后是gather,将每个并行设备中的output再合并到一起

### scatter散播函数
```python
def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        return scatter_map(inputs)
    finally:
        scatter_map = None

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    r"""Scatter with support for kwargs dictionary"""
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs
```

**对于list, tuple, dict类型的数据都不分割，scatter数据，予以保留然后转换成list类型**

**对于Tensor类型的数据都送入Scatter.apply()函数中，这个函数最终将Tensor送入torch._C._scatter()函数（由C实现）**

**但是在scatter函数注释写的input类型是tensor，所以input推荐使用tensor类型**

### gather聚合函数
```python
def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError('All dicts must have the same number of keys')
            return type(out)(((k, gather_map([d[k] for d in outputs]))
                              for k in out))
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        return gather_map(outputs)
    finally:
        gather_map = None
```
对于Gather.apply()送入的是一组tensor：

tensors (Iterable[Tensor]): iterable of tensors to gather.

**这里有一个额外的支持！！支持dict类型**

因为有可能model有多项输出，因为存在model有分支的情况，所以pytorch并行支持的多项输出存放在dict里

例如对于outputs是这样的：
```python
outputs = [{'w':[1,2,3],'v':[1,3,5]},{'w':[4,5,6],'v':[2,4,6]}]
```
那么首先对于out也就是,sample_0
```python
out = {'w':[1,2,3],'v':[1,3,5]}
```
接下来iter每个key，然后对于每个sample都把这个key的tensor聚合在一起:
```python
((k, gather_map([d[k] for d in outputs])) for k in out)

类似列表生成器，得到一个generator对象
```
通过type(out)(generator)，可以将这个重新变回dict，所以保证还是一个dict，但是里面的tensor已经经过并行合并了

**所以如果模型有多个输出，而且大小不等，不能合并成一个tensor的情况下，这里推荐输出的类型是dict**
