## tf.placeholder函数
***tf.placeholder(dtype, shape=None, name=None)***

placeholder，占位符，在tensorflow中类似于函数参数，运行时必须传入值。
```
    dtype：数据类型。常用的是tf.float32,tf.float64等数值类型。
    shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定。
    name：名称。
```
一般与sess.run， feed_dict搭配使用
