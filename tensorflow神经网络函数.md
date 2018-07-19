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
### tf.truncated_normal
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
    
### tf.random_normal
***tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)***

从正态分布中输出随机值。
参数:

    shape: 一维的张量，也是输出的张量。
    mean: 正态分布的均值。
    stddev: 正态分布的标准差。
    dtype: 输出的类型。
    seed: 一个整数，当设置之后，每次生成的随机数都一样。
    name: 操作的名字。

代码
```python
a = tf.Variable(tf.random_normal([2,2],seed=1))
b = tf.Variable(tf.truncated_normal([2,2],seed=2))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a))
    print(sess.run(b))
```
```
输出：
[[-0.81131822  1.48459876]
 [ 0.06532937 -2.44270396]]
[[-0.85811085 -0.19662298]
 [ 0.13895047 -1.22127688]]
```

https://blog.csdn.net/zj360202/article/details/70243127
https://blog.csdn.net/u013713117/article/details/65446361
