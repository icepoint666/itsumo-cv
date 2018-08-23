# zero_grad

有两种方式直接把模型的参数梯度设成0：
```python
model.zero_grad()
optimizer.zero_grad() # 当optimizer=optim.Optimizer(model.parameters())时，两者等效
```
如果想要把某一Variable的梯度置为0，只需用以下语句：
`Variable.grad.data.zero_()`
```
# Zero the gradients before running the backward pass.
    model.zero_grad()

# Before the backward pass, use the optimizer object to zero all of the
# gradients for the variables it will update (which are the learnable weights
# of the model)
    optimizer.zero_grad()
```

https://blog.csdn.net/qq_34690929/article/details/79934843
