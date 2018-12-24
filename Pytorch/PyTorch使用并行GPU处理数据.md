# Pytorch使用并行GPU处理数据
将Module放在GPU上运行也十分简单，只需两步：
```python
model = model.cuda() # 将模型的所有参数转存到GPU
input.cuda() # 将输入数据也放置到GPU上
```
至于如何在多个GPU上并行计算，PyTorch也提供了两个函数，可实现简单高效的并行GPU计算。
```python
nn.parallel.data_parallel(module, inputs, device_ids=None, output_device=None, dim=0, module_kwargs=None)
```
```python
class torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)
```

二者的参数十分相似，通过device_ids参数可以指定在哪些GPU上进行优化，output_device指定输出到哪个GPU上。

唯一的不同就在于前者直接利用多GPU并行计算得出结果，而后者则返回一个新的module，能够自动在多GPU上进行并行加速。

```python
# method 1
new_net = nn.DataParallel(net, device_ids=[0, 1])
output = new_net(input)
# method 2
output = nn.parallel.data_parallel(net, input, device_ids=[0, 1])
```

**推荐：一般的可以直接在nn.module类中的forward函数中写并行代码**

```python
def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else: 
            output = self.net(input)
        return output 
```


https://blog.csdn.net/u013063099/article/details/79579407
