# Tensorflow加载pretrained模型
### 1.介绍tf.Graph
tf.Graph包含两类相关信息：

**图结构.** 图的节点和边缘，指明了各个指令组合在一起的方式，但不规定它们的使用方式。图结构与汇编代码类似：检查图结构可以传达一些有用的信息，但它不包含源代码传达的的所有有用上下文。

**图集合.** TensorFlow提供了一种通用机制，以便在tf.Graph中存储元数据集合。

tf.add_to_collection函数允许您将对象列表与一个键相关联(其中tf.GraphKeys定义了部分标准键)

tf.get_collection则允许您查询与键关联的所有对象。

TensorFlow库的许多组成部分会使用它：

例如，当您创建tf.Variable时，系统会默认将其添加到表示“全局变量(tf.global_variables)”和“可训练变量(tf.trainable_variables)”的集合中。当您后续创建tf.train.Saver或tf.train.Optimizer时，这些集合中的变量将用作默认参数。

也就是说，在创建图的过程中，TensorFlow的Python底层会自动用一些collection对op进行归类，方便之后的调用。这部分collection的名字被称为tf.GraphKeys，可以用来获取不同类型的op。当然，我们也可以自定义collection来收集op。

#### 常见GraphKeys
- GLOBAL_VARIABLES: 该collection默认加入所有的Variable对象，并且在分布式环境中共享。一般来说，TRAINABLE_VARIABLES包含在MODEL_VARIABLES中，MODEL_VARIABLES包含在GLOBAL_VARIABLES中。
- LOCAL_VARIABLES: 与GLOBAL_VARIABLES不同的是，它只包含本机器上的Variable，即不能在分布式环境中共享。
- MODEL_VARIABLES: 顾名思义，模型中的变量，在构建模型中，所有用于正向传递的Variable都将添加到这里。
- TRAINALBEL_VARIABLES: 所有用于反向传递的Variable，即可训练(可以被optimizer优化，进行参数更新)的变量。
- SUMMARIES: 跟Tensorboard相关，这里的Variable都由tf.summary建立并将用于可视化。
- QUEUE_RUNNERS: the QueueRunner objects that are used to produce input for a computation.
- MOVING_AVERAGE_VARIABLES: the subset of Variable objects that will also keep moving averages.
- REGULARIZATION_LOSSES: regularization losses collected during graph construction.

### 2.test时导入预训练模型Demo
```python
    with tf.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32) # 输入数据:input_image
        output = model.build_server_graph(input_image)           # 建立计算图
        output = (output + 1.) * 127.5 
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # 获取所有收集到的op
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.contrib.framework.load_variable(args.checkpoint_dir, from_name) # 获取所有model variables
            assign_ops.append(tf.assign(var, var_value))          # 将收集到的op赋值,并保存到一个list中,传入sess.run()运行
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
```
### References:

1. https://github.com/JiahuiYu/generative_inpainting/blob/master/test.py
2. https://blog.csdn.net/hustqb/article/details/80398934
