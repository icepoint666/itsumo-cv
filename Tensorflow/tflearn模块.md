# tflearn
安装tflearn

在安装tflearn之前，除了要确保tensorflow成功安装外，还要安装额外的两个依赖包，否则运行时会发生错误（即使可以成功安装）：
```
pip install scipy h5py
```

之后使用以下语句进行安装（根据个人喜好自己选择安装方式，二选一）：
```
# 安装最新beta版
$ pip install git+https://github.com/tflearn/tflearn.git

(需先安装git，如果没有安装，使用sudo apt-get install git命令进行安装)

# 安装最近的稳定版
$ pip install tflearn
```

最后进入python环境，import tflearn 若成功导入，则安装成功。如果出错，多数是因为之前的tensorflow配置错误，抑或是python版本出错，请参阅上边的教程，确保正确安装。

使用：
```python
import tflearn
```
