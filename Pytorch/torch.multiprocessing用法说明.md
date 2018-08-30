# torch.multiprocessing
封装了multiprocessing模块。用于在相同数据的不同进程中共享视图。

一旦张量或者存储被移动到共享单元(见share_memory_()),它可以不需要任何其他复制操作的发送到其他的进程中。

这个API与原始模型完全兼容，为了让张量通过队列或者其他机制共享，移动到内存中，我们可以由原来的import multiprocessing改为import torch.multiprocessing。

由于API的相似性，我们没有记录这个软件包的大部分内容，我们建议您参考原始模块的非常好的文档。

warning： 如果主要的进程突然退出(例如，因为输入信号)，Python中的multiprocessing有时会不能清理他的子节点。

这是一个已知的警告，所以如果您在中断解释器后看到任何资源泄漏，这可能意味着这刚刚发生在您身上。
### Strategy management
```python
torch.multiprocessing.get_all_sharing_strategies()
```
返回一组由当前系统所支持的共享策略
```python
torch.multiprocessing.get_sharing_strategy()
```
返回当前策略共享CPU中的张量。
```python
torch.multiprocessing.set_sharing_strategy(new_strategy)
```
设置共享CPU张量的策略

参数: new_strategy(str)-被选中策略的名字。应当是get_all_sharing_strategies()中值当中的一个。

### Sharing CUDA tensors

共享CUDA张量进程只支持Python3，使用spawn或者forkserver开始方法。

Python2中的multiprocessing只能使用fork创建子进程，并且不被CUDA支持。

warning： CUDA API要求导出到其他进程的分配一直保持有效，只要它们被使用。

你应该小心，确保您共享的CUDA张量不要超出范围。

这不应该是共享模型参数的问题，但传递其他类型的数据应该小心。请注意，此限制不适用于共享CPU内存。

https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-multiprocessing/
