注意一下两种用法中，上面这种会报错
### A
```python
class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc_1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        return F.log_softmax(self.fc_1(output), dim=-1)
        
    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
```
### B
```python
class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.fc_1 = nn.Linear(2, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = nn.PReLU(output)
        return F.log_softmax(self.fc_1(output), dim=-1)
        
    def get_embedding(self, x):
        return nn.PReLU(self.embedding_net(x))
```
运行第二种报错：
```
AttributeError: 'PReLU' object has no attribute 'dim'.
```

