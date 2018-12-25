# tf.device()指定运行设备

在TensorFlow中，模型可以在本地的GPU和CPU中运行，用户可以指定模型运行的设备。

通常，如果你的TensorFlow版本是GPU版本的，而且你的电脑上配置有符合条件的显卡，那么在不做任何配置的情况下，模型是默认运行在显卡下的。

如果需要切换成CPU运算，可以调用`tf.device(device_name)`函数，其中device_name格式如`/cpu:`

0其中的0表示设备号，TF不区分CPU的设备号，设置为0即可。

GPU区分设备号`\gpu:0`和`\gpu:1`表示两张不同的显卡。 

在一些情况下，我们即使是在GPU下跑模型，也会将部分Tensor储存在内存里，因为这个Tensor可能太大了，显存不够放，相比于显存，内存一般大多了，

于是这个时候就常常人为指定为CPU设备。这种形式我们在一些代码中能见到。如：
```python
with tf.device('/cpu:0'):
    build_CNN() # 此时，这个CNN的Tensor是储存在内存里的，而非显存里。
```

https://blog.csdn.net/LoseInVain/article/details/78814091 
