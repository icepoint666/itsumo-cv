# tf.nn.conv2d

tf.nn.conv2d是TensorFlow里面实现卷积的函数，参考文档对它的介绍并不是很详细，实际上这是搭建卷积神经网络比较核心的一个方法，非常重要

***tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)***
```
除去name参数用以指定该操作的name，与方法有关的一共五个参数：

第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一

第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维

第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4

第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）

第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true

结果返回一个Tensor，这个输出，就是我们常说的feature map，shape仍然是[batch, height, width, channels]这种形式。
```

那么TensorFlow的卷积具体是怎样实现的呢，用一些例子去解释它：

1.考虑一种最简单的情况，现在有一张3×3单通道的图像（对应的shape：[1，3，3，1]），用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，最后会得到一张3×3的feature map

2.增加图片的通道数，使用一张3×3五通道的图像（对应的shape：[1，3，3，5]），用一个1×1的卷积核（对应的shape：[1，1，1，1]）去做卷积，仍然是一张3×3的feature map，这就相当于每一个像素点，卷积核都与该像素点的每一个通道做卷积。
```python
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([1,1,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
```
3.把卷积核扩大，现在用3×3的卷积核做卷积，最后的输出是一个值，相当于情况2的feature map所有像素点的值求和
```python
input = tf.Variable(tf.random_normal([1,3,3,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
```
4.使用更大的图片将情况2的图片扩大到5×5，仍然是3×3的卷积核，令步长为1，输出3×3的feature map
```python
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='VALID')
```
注意我们可以把这种情况看成情况2和情况3的中间状态，卷积核以步长1滑动遍历全图，以下x表示的位置，表示卷积核停留的位置，每停留一个，输出feature map的一个像素
```
.....
.xxx.
.xxx.
.xxx.
.....
```
5.上面我们一直令参数padding的值为‘VALID’，当其为‘SAME’时，表示卷积核可以停留在图像边缘，如下，输出5×5的feature map
```python
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,1]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
```
```
xxxxx
xxxxx
xxxxx
xxxxx
xxxxx
```
6.如果卷积核有多个
```python
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME')
```
此时输出7张5×5的feature map

7.步长不为1的情况，文档里说了对于图片，因为只有两维，通常strides取[1，stride，stride，1]
```python
input = tf.Variable(tf.random_normal([1,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
```
此时，输出7张3×3的feature map
```
x.x.x
.....
x.x.x
.....
x.x.x
```
8.如果batch值不为1，同时输入10张图
```python 
input = tf.Variable(tf.random_normal([10,5,5,5]))
filter = tf.Variable(tf.random_normal([3,3,5,7]))

op = tf.nn.conv2d(input, filter, strides=[1, 2, 2, 1], padding='SAME')
```
每张图，都有7张3×3的feature map，输出的shape就是[10，3，3，7]

最后，把程序总结一下：

```python
import tensorflow as tf

# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
# 除去name参数用以指定该操作的name，与方法有关的一共五个参数：
#
# 第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#
# 第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
#
# 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4
#
# 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式（后面会介绍）
#
# 第五个参数：use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
#
# 结果返回一个Tensor，这个输出，就是我们常说的feature map

oplist=[]
# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([1 ,1 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 2"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 3, 3, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 3"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))

op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='VALID')
oplist.append([op2, "case 4"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,1]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 5"])

# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,1,1,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 6"])


# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([1, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,2,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 7"])


# [batch, in_height, in_width, in_channels]
input_arg  = tf.Variable(tf.ones([4, 5, 5, 5]))
# [filter_height, filter_width, in_channels, out_channels]
filter_arg = tf.Variable(tf.ones([3 ,3 , 5 ,7]))
op2 = tf.nn.conv2d(input_arg, filter_arg, strides=[1,2,2,1], use_cudnn_on_gpu=False, padding='SAME')
oplist.append([op2, "case 8"])

with tf.Session() as a_sess:
    a_sess.run(tf.global_variables_initializer())
    for aop in oplist:
        print("----------{}---------".format(aop[1]))
        print(a_sess.run(aop[0]))
        print('---------------------\n\n')

```

 

结果是这样的：

 
```
----------case 2---------
[[[[ 5.]
[ 5.]
[ 5.]]

[[ 5.]
[ 5.]
[ 5.]]

[[ 5.]
[ 5.]
[ 5.]]]]
---------------------


----------case 3---------
[[[[ 45.]]]]
---------------------


----------case 4---------
[[[[ 45.]
[ 45.]
[ 45.]]

[[ 45.]
[ 45.]
[ 45.]]

[[ 45.]
[ 45.]
[ 45.]]]]
---------------------


----------case 5---------
[[[[ 20.]
[ 30.]
[ 30.]
[ 30.]
[ 20.]]

[[ 30.]
[ 45.]
[ 45.]
[ 45.]
[ 30.]]

[[ 30.]
[ 45.]
[ 45.]
[ 45.]
[ 30.]]

[[ 30.]
[ 45.]
[ 45.]
[ 45.]
[ 30.]]

[[ 20.]
[ 30.]
[ 30.]
[ 30.]
[ 20.]]]]
---------------------


----------case 6---------
[[[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]]]
---------------------


----------case 7---------
[[[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]]]
---------------------


----------case 8---------
[[[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]]


[[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]]


[[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]]


[[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]

[[ 30. 30. 30. 30. 30. 30. 30.]
[ 45. 45. 45. 45. 45. 45. 45.]
[ 30. 30. 30. 30. 30. 30. 30.]]

[[ 20. 20. 20. 20. 20. 20. 20.]
[ 30. 30. 30. 30. 30. 30. 30.]
[ 20. 20. 20. 20. 20. 20. 20.]]]]
---------------------
```

http://www.cnblogs.com/welhzh/p/6607581.html
