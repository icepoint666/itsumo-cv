# tf.reshape()函数
```
tf.reshape(tensor,shape,name=None)1
```
函数的作用是将tensor变换为参数shape形式，其中的shape为一个列表形式，特殊的是列表可以实现逆序的遍历，即list(-1).-1所代表的含义是我们不用亲自去指定这一维的大小，函数会自动进行计算，但是列表中只能存在一个-1。（如果存在多个-1，就是一个存在多解的方程） 
下面就说一下reshape是如何进行矩阵的变换的，其简单的流程就是： 
将矩阵t变换为一维矩阵，然后再对矩阵的形式进行更改就好了，具体的流程如下：
```
reshape(t,shape) =>reshape(t,[-1]) =>reshape(t,shape)
```
```python
output = tf.reshape(inputs, [-1, 3, 64, 64])
```
