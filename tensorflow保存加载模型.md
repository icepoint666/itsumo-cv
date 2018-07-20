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
