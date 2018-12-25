# tf.split()函数

tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，如果是0就表示对第0维度进行切割。

num_split就是切割的数量，如果是2就表示输入张量被切成2份，每一份是一个列表。
```python
import tensorflow as tf
import numpy as np;
A = [[1,2,3],[4,5,6]
x = tf.split(1, 3, A)
with tf.Session() as sess:
    c = sess.run(x)	
    for ele in c:		
        print ele

[[1]
 [4]]
[[2]
 [5]]
[[3]
 [6]]
```

注意：

版本不同会有改动的，也就是函数用法会不同，注意一下子

上面的用法是tensorflow 0.*的

tensorflow 1.*也有新的用法

**使用：**
```python
    if tf.__version__.startswith('1.'):
        split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
    else:
        split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)

```
