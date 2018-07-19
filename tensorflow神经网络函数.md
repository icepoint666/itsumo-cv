# tensorflow神经网络函数
### tf.placeholder函数

***tf.placeholder(dtype, shape=None, name=None)***


此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值

参数：
```
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
    name：名称。
```
示例：
```python
    x = tf.placeholder(tf.float32, shape=(1024, 1024))
    y = tf.matmul(x, x)
     
    with tf.Session() as sess:
      print(sess.run(y))  # ERROR: 此处x还没有赋值.
     
      rand_array = np.random.rand(1024, 1024)
      print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```
返回：Tensor 类型
### tf.truncated_normal函数
***tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)***

从截断的正态分布中输出随机值。
生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择。

在正态分布的曲线中，横轴区间（μ-σ，μ+σ）内的面积为68.268949%。
横轴区间（μ-2σ，μ+2σ）内的面积为95.449974%。
横轴区间（μ-3σ，μ+3σ）内的面积为99.730020%。
X落在（μ-3σ，μ+3σ）以外的概率小于千分之三，在实际问题中常认为相应的事件是不会发生的，基本上可以把区间（μ-3σ，μ+3σ）看作是随机变量X实际可能的取值区间，这称之为正态分布的“3σ”原则。
在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。这样保证了生成的值都在均值附近。

参数:

    shape: 一维的张量，也是输出的张量。
    mean: 正态分布的均值。
    stddev: 正态分布的标准差。
    dtype: 输出的类型。
    seed: 一个整数，当设置之后，每次生成的随机数都一样。
    name: 操作的名字。
    
### tf.random_normal函数
***tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)***

从正态分布中输出随机值。
参数:

    shape: 一维的张量，也是输出的张量。
    mean: 正态分布的均值。
    stddev: 正态分布的标准差。
    dtype:p.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])

inputs_pad = tf.pad(inputs,pad_mat)
```
其中pad_mat每行表示每一维的两个方向上的padding量

示例1：
```python
t=[[2,3,4],[5,6,7]]
paddings=[[1,1],[2,2]]
mode="CONSTANT"
sess.run(tf.pad(t, paddings, "CONSTANT"))
```
输出结果：
```
array([[0, 0, 0, 0, 0, 0, 0],
          [0, 0, 2, 3, 4, 0, 0],
          [0, 0, 5, 6, 7, 0, 0],
          [0, 0, 0, 0, 0, 0, 0]], dtype=int32)
```
可以看到，上，下，左，右分别填充了1,1,2,2行刚好和paddings=[[1,1],[2,2]]相等，零填充

示例2：
```python
t=[[2,3,4],[5,6,7]]
paddings=[[1,2],[2,3]]
mode="CONSTANT"
sess.run(tf.pad(t,paddings,"CONSTANT"))
```
输出结果：
```
array([[0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 2, 3, 4, 0, 0, 0],
          [0, 0, 5, 6, 7, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)
```
可以看到，上，下，左，右分别填充啦1,2,2,3行刚好和paddings=[[1,2],[2,3]]相等，零填充

示例3:
```python
t=[[2,3,4],[5,6,7]]
paddings=[[1,1],[2,2]]
mode='REFLECT'
sess.run(tf.pad(t,paddings,"REFLECT"))
```
输出结果：
```
array([[7, 6, 5, 6, 7, 6, 5],
           [4, 3, 2, 3, 4, 3, 2],
           [7, 6, 5, 6, 7, 6, 5],
           [4, 3, 2, 3, 4, 3, 2]], dtype=int32)
```
可以看到，上下左右的值进行了映射填充，上下值填充的顺序和t是相反的，左右值只是进行顺序补齐

示例4：
```python
t=[[2,3,4],[5,6,7]]
paddings=[[1,1],[2,2]]
mode='SYMMETRIC'
sess.run(tf.pad(t,paddings,"SYMMETRIC"))
```
输出结果：
```
array([[3, 2, 2, 3, 4, 4, 3],
          [3, 2, 2, 3, 4, 4, 3],
          [6, 5, 5, 6, 7, 7, 6],
          [6, 5, 5, 6, 7, 7, 6]], dtype=int32)
```
可以看到，上下左右的值进行了对称填充，上下值是按照t相同顺序填充的，左右值只是进行对称补齐

### tf.nn.bias_add
***tf.nn,bias_add(value, bias, name = None)***
这个函数的作用是将bias加到value上

这个操作可以看成tf.add的特例，其中bias是一维的。

输入参数：
```
value: 输入一个Tensor， 其类型只能是float, double, int64, int32, int16, int8, uint8, complex64
bias: 一个一维的Tensor数据维度必须和Tensor的最后一维相同，数据类型必须和value的相同
name： (可选)为这个操作取一个名字
```
输出：
```
一个Tensor，数据类型必须和value相同。
```
示例：
```python
import tensorflow as tf
 
a=tf.constant([[1,1],[2,2],[3,3]],dtype=tf.float32)
b=tf.constant([1,-1],dtype=tf.float32)
c=tf.constant([1],dtype=tf.float32)
 
with tf.Session() as sess:
    print('bias_add:')
    print(sess.run(tf.nn.bias_add(a, b)))
    #执行下面语句错误
    #print(sess.run(tf.nn.bias_add(a, c)))
 
    print('add:')
    print(sess.run(tf.add(a, c)))
```
输出结果：
```
bias_add:
[[ 2. 0.]
[ 3. 1.]
[ 4. 2.]]
add:
[[ 2. 2.]
[ 3. 3.]
[ 4. 4.]]
```

https://blog.csdn.net/zj360202/article/details/70243127

https://blog.csdn.net/u013713117/article/details/65446361

https://blog.csdn.net/zhang_bei_qing/article/details/75090203

https://blog.csdn.net/weixin_38698649/article/details/80100737
