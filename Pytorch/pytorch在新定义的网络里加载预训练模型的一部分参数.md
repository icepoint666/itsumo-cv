# pytorch在新定义的网络里记载预训练模型的一部分参数

**对于pytorch的保存的模型`netG.pth`，其实就是`OrderedDict()`**

输出看一下：
```python
>>> pretrained_net = torch.load("netG.pth")
>>> print(pretrained_net)

('model.dmap_branch_post.4.conv_block_stream1.5.bias', tensor([ 1.1563e-02,  1.9707e-02, -3.9119e-03, ..., 4.6473e-03])), 
('model.dmap_branch_post.4.conv_block_stream2.1.weight', tensor([[[[-7.1143e-03,  2.0464e-02,  9.5958e-03],...]]]),
...

```
类型：
```python
>>> print(type(pretrained_net))
<class 'collections.OrderedDict'>
```

输出keys()：
```python
for idx, key in enumerate(pretrained_net):
    print(idx, key)
    
0 model.dmap_branch_stream1_down.1.weight
1 model.dmap_branch_stream1_down.1.bias
2 model.dmap_branch_stream1_down.4.weight
3 model.dmap_branch_stream1_down.4.bias
4 model.dmap_branch_stream1_down.7.weight
5 model.dmap_branch_stream1_down.7.bias
6 model.dmap_branch_stream2_down.1.weight
...
102 model.dmap_branch_post.5.conv_block_stream2.5.weight
103 model.dmap_branch_post.5.conv_block_stream2.5.bias
104 model.dmap_branch_stream1_up.0.weight
105 model.dmap_branch_stream1_up.0.bias
106 model.dmap_branch_stream1_up.3.weight
107 model.dmap_branch_stream1_up.3.bias
108 model.dmap_branch_stream1_up.7.weight
109 model.dmap_branch_stream1_up.7.bias

```

### strict=False选项

**这里你假如想重新写一个网络，但是不用上采样层了，不过你还需要从原来的整体预训练模型中导入参数**

**这里解决办法就是`load_state_dict`函数有一个`strict=False`选项**
**作用就是：对于新定义的网络，加载模型时，它直接忽略那些没有的dict，有相同的就复制，没有就直接放弃赋值！（根据key来查找）**
```python
new_network.load_state_dict(pretrained_net, strict=False)
```

保存的Dict是按照net.属性.weight来存储的。如果这个属性是一个Sequential，我们可以类似这样`net.seqConvs.0.weight`来获得。

当然在定义的类中，拿到Sequential的某一层用[], 比如`self.seqConvs[0].weight`.

strict=False是没有那么智能，遵循有相同的key则赋值，否则直接丢弃。

验证新导入的参数与预训练模型的参数是否一致
```python
pretrained_net = torch.load("netG.pth")
new_network.load_state_dict(pretrained_net, strict=False)
print(new_network.state_dict())
print(pretrained_net)
```

**这里有一个模型某层的权重赋值示例**
```python
def _initialize_weights_from_net(self):
        save_path = 't.pth'
        print('Successfully load model '+save_path)
        # First load the net.
        pretrained_net = Net_old()
        pretrained_net_dict = torch.load(save_path)
        # load params
        pretrained_net.load_state_dict(pretrained_net_dict)

        new_convs = self.get_convs()

        cnt = 0
        # Because sequential is a generator.
                for i, name in enumerate(pretrained_net.nets):
            if isinstance(name, torch.nn.Conv2d):
                print('Assign weight of pretrained model layer : ', name, ' to layer: ', new_convs[cnt])
                new_convs[cnt].weight.data = name.weight.data
                new_convs[cnt].bias.data = name.bias.data
                cnt += 1

    def get_convs(self):
        return [self.conv1, self.conv2, self.conv3]
```
参考链接：

https://blog.csdn.net/hungryof/article/details/81364487
