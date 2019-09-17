# pytorch模型保存时存在的一些问题

### 1. pytorch保存模型存在gpu/cpu类型

一般保存代码：
```python
torch.save(model.state_dict(),'./checkpoints/epoch_'+str(epoch)+".pth")
```
但是如果model是在gpu设备上的tensor，那么其实保存的也是gpu类型的模型

会导致在下次加载时，直接把模型加载成gpu设备上的torch.cuda.Tensor类型，即使你前面定义的是cpu的模型，例如下面这样
```python
model = FlowSD()
model.load_state_dict(torch.load(model_path))
```
**所以一般建议将模型保存为cpu型的**
```python
torch.save(model.cpu().state_dict(), './checkpoints/epoch_'+str(epoch)+".pth")
```

### 2. torch.save保存模型时使用cpu()会修改model的类型

如果之前model类型是gpu类型，使用下面的语句后
```python
torch.save(model.cpu().state_dict(), './checkpoints/epoch_'+str(epoch)+".pth")
```
model类型会变成cpu类型，对于之后的迭代可能会报错，所以这里建议保存后，加一句话，将它再转回gpu上
```python
torch.save(network.cpu().state_dict(), save_path)
if len(gpu_ids) and torch.cuda.is_available():
    network.cuda(gpu_ids[0])
```
**但是这样做可能会造成现存增大，目前还为搞清可能的原因**
