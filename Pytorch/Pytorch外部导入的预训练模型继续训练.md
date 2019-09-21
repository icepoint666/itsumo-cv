# Pytorch外部导入的预训练模型继续训练

正常optimizer:
```python
self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
```
**使用add_param_group函数可以完成这样的操作,但是只能添加某些层**
详见：https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py

**其实可以直接这样**
```python
self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
self.optimizer_mapG = torch.optim.Adam(self.mapG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
self.optimizer_flowG = torch.optim.Adam(self.flowG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
```
