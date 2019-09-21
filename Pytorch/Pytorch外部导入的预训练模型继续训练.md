# Pytorch外部导入的预训练模型继续训练

正常optimizer:
```python
self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
```
**使用add_param_group函数可以完成这样的操作**
```python
self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
self.optimizer_G.add_param_group(self.mapG.parameters())
self.optimizer_G.add_param_group(self.flowG.parameters())
```

详见：https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py
