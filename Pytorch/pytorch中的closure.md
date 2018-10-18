# pytorch中的closure

optimizer.step(closure)

**一些优化算法，如共轭梯度和LBFGS需要重新评估目标函数多次，所以你必须传递一个closure以重新计算模型。 closure必须清除梯度，计算并返回损失**
```python
for input, target in dataset:
    def closure():
        optimizer.zero_grad()
        output = model(input)
        loss = loss_fn(output, target)
        loss.backward()
        return loss
    optimizer.step(closure)
```
