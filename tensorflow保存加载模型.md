# tensorflow保存加载模型

## 1、保存模型
```python
# 首先定义saver类
saver = tf.train.Saver(max_to_keep=4)
# 定义会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(300):
        if epoch % 10 == 0:
            # 保存模型
            saver.save(sess, "model/my-model", global_step=epoch)
            print "save the model"

        # 训练
        sess.run(train_step)
```
这里saver.sess函数中"model/my-model"，表示存储路径在/model路径下

my-model表示存储模型名，也就是存储模型文件的前缀

假如epoch是160，那么存储的文件如下：
```
/model
  my-model-160.data-00000-of-00001
  my-model-160.index
  my-model-160.mata
  checkpoint
```
注意点：
- 创建saver时，可以指定需要存储的tensor，如果没有指定，则全部保存。

- 创建saver时，可以指定保存的模型个数，利用max_to_keep=4，则最终会保存4个模型（例如保存160、170、180、190step共4个模型）。

- saver.save()函数里面可以设定global_step，说明是哪一步保存的模型。

- 程序结束后，会生成四个文件：存储网络结构.meta、存储训练好的参数.data和.index、记录最新的模型checkpoint。

## 2、加载模型
```python
def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/my-model-290.meta')
        saver.restore(sess, tf.train.latest_checkpoint("model/"))
```

注意点：

- 首先import_meta_graph，这里填的名字meta文件的名字。然后restore时，是检查checkpoint，所以只填到checkpoint所在的路径下即可，不需要填checkpoint，不然会报错“ValueError: Can’t load save_path when it is None.”。

## 3、线性拟合例子
```python
import tensorflow as tf
import numpy as np

def train_model():

    # prepare the data
    x_data = np.random.rand(100).astype(np.float32)
    print x_data
    y_data = x_data * 0.1 + 0.2
    print y_data

    # define the weights
    W = tf.Variable(tf.random_uniform([1], -20.0, 20.0), dtype=tf.float32, name='w')
    b = tf.Variable(tf.random_uniform([1], -10.0, 10.0), dtype=tf.float32, name='b')
    y = W * x_data + b

    # define the loss
    loss = tf.reduce_mean(tf.square(y - y_data))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # save model
    saver = tf.train.Saver(max_to_keep=4)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print "------------------------------------------------------"
        print "before the train, the W is %6f, the b is %6f" % (sess.run(W), sess.run(b))

        for epoch in range(300):
            if epoch % 10 == 0:
                print "------------------------------------------------------"
                print ("after epoch %d, the loss is %6f" % (epoch, sess.run(loss)))
                print ("the W is %f, the b is %f" % (sess.run(W), sess.run(b)))
                saver.save(sess, "model/my-model", global_step=epoch)
                print "save the model"
            sess.run(train_step)
        print "------------------------------------------------------"

def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('model/my-model-290.meta')
        saver.restore(sess, tf.train.latest_checkpoint("model/"))
        print sess.run('w:0')
        print sess.run('b:0')

train_model()
load_model()
```
    首先定义了y=ax+b的线性关系，a=0.1，b=0.2，然后给定训练数据集，w是-20.0到20.0之间的任意数，b是-10.0到10.0之间的任意数。

    然后定义损失函数，定义随机梯度下降训练器。

    定义saver后进入训练阶段，边训练边保存模型。并输出中间的训练loss，w和b。可以看到w和b在逐步接近我们设定的0.1和0.2。

    在load_model函数中，我们首先利用第2小节中的方法加载模型，然后就可以根据模型中权值的名字，打印其结果。

注意：

这里说明一点，如何知道tensor的名字，最好是定义tensor的时候就指定名字，如上面代码中的name='w'，如果你没有定义name，tensorflow也会设置name，只不过这个name就是根据你的tensor或者操作的性质，像上面的w，这是“Variable:0”，loss则是“Mean:0”。所以最好还是自己定义好name。

## 4.知识点

1、.meta文件：一个协议缓冲，保存tensorflow中完整的graph、variables、operation、collection。

2、checkpoint文件：一个二进制文件，包含了weights, biases, gradients和其他variables的值。但是0.11版本后的都修改了，用.data和.index保存值，用checkpoint记录最新的记录。

3、在进行保存时，因为meta中保存的模型的graph，这个是一样的，只需保存一次就可以，所以可以设置saver.save(sess, 'my-model', write_meta_graph=False)即可。

4、如果想设置每多长时间保存一次，可以设置saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)，这个是每2个小时保存一次。

5、如果不想保存所有变量，可以在创建saver实例时，指定保存的变量，可以以list或者dict的类型保存。如：
```python
w1 = tf.Variable(tf.random_normal(shape=[2]), name='w1')
w2 = tf.Variable(tf.random_normal(shape=[5]), name='w2')
saver = tf.train.Saver([w1,w2])
```
