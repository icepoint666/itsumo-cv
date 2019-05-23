# pytorch中将一个tensor在任意维度上进行shuffle

**`tensorflow`有单独这个洗牌tensor操作的实现：**
```python
b = tf.random_shuffle(a)
```
将a里面单维度内的元素进行随机洗牌

**对应`pytorch`里面的替代操作是:**

假如元素第一维的维度是100，那么使用randperm函数可以在这生成0-99的随机数，利用python tensor的切片功能可以得到这样一个shuffle后的tensor
```python
x = torch.randn(100,3,32,32)
x_perm = x[torch.randperm(100)]
```
You can combine the tensors using stack if they’re in a python list. You can also use something like `x.index_select(0, torch.randperm(100)` .
