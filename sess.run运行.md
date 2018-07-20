# tensorflow运行
## tf.placeholder函数
***tf.placeholder(dtype, shape=None, name=None)***

placeholder，占位符，在tensorflow中类似于函数参数，运行时必须传入值。
```
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型。
    shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定。
    name：名称。
```
一般与sess.run， feed_dict搭配使用
## sess.run函数
这里 self.sess.run(）函数是执行一个会话，第一个参数是图的输出节点，第二个参数图的输入节点。

示例：
```python
self.sess.run([d_optim, self.d_sum], feed_dict={ self.images: batch_images, self.z: batch_z })，
```

上面的会话会根据输出节点d_optim, self.d_sum在图中找到最初的输入节点。

d_optim———>d_loss——->D_logits, D_logits_。

其中D_logits的输入是self.images， D_logits_的输入是self.z。因此这里run的第二个参数应该为{self.images，self.z}。

但是self.images，self.z只是个用placeholder定义的占位符，因此需要指定实际的输入。所以，这里用feed_dict指定了个字典，key值为self.images的占位符对应的值为batch_images，即加载的真实图片数据。key值为self.z的占位符对应的值为batch_z，即噪音数据。

这里看一下self.images，self.z的定义，均是用placeholder生成的占位符。
```python
self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                name='real_images')

self.z = tf.placeholder(tf.float32, [None, self.z_dim],name='z') 
```
![](pics/sess_run_1.png) 
![](pics/sess_run_2.png) 

https://blog.csdn.net/cc1949/article/details/78364615?locationNum=4&fps=1
