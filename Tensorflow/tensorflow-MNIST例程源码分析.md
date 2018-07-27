# tensorflow-MNIST例程源码分析
运行 MNIST 例程的命令为：
```shell
# python -m tensorflow.models.image.mnist.convolutional
```
对应文件为 /usr/lib/python2.7/site-packages/tensorflow/models/image/mnist/convolutional.py
打开例程源码：
```python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# 数据源
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
# 工作目录，存放下载的数据
WORK_DIRECTORY = 'data'
# MNIST 数据集特征： 
#     图像尺寸 28x28 
IMAGE_SIZE = 28
#     黑白图像
NUM_CHANNELS = 1
#     像素值0~255 
PIXEL_DEPTH = 255
#     标签分10个类别
NUM_LABELS = 10
#     验证集共 5000 个样本
VALIDATION_SIZE = 5000  
# 随机数种子，可设为 None 表示真的随机
SEED = 66478 
# 批处理大小为64
BATCH_SIZE = 64
# 数据全集一共过10遍网络
NUM_EPOCHS = 10
# 验证集批处理大小也是64
EVAL_BATCH_SIZE = 64
# 验证时间间隔，每训练100个批处理，做一次评估
EVAL_FREQUENCY = 100  


tf.app.flags.DEFINE_boolean("self_test", False, "True if running a self test.")
FLAGS = tf.app.flags.FLAGS

# 如果下载过了数据，就不再重复下载
def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.Size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

# 抽取数据，变为 4维张量[图像索引，y, x, c]
# 去均值、做归一化，范围变到[-0.5, 0.5]
def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
return data

# 抽取图像标签
def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

# 假数据，用于功能自测
def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels
# 计算分类错误率
def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
predictions.shape[0])




# 主函数
def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # 下载数据
    train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # 载入数据到numpy
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    # 产生评测集
    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]
    num_epochs = NUM_EPOCHS
  train_size = train_labels.shape[0]

# 训练样本和标签将从这里送入网络。
# 每训练迭代步，占位符节点将被送入一个批处理数据
# 训练数据节点
  train_data_node = tf.placeholder(
      tf.float32,
shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
# 训练标签节点
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
# 评测数据节点
  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

# 下面这些变量是网络的可训练权值
# conv1 权值维度为 32 x channels x 5 x 5, 32 为特征图数目
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.1,
                          seed=SEED))
# conv1 偏置
  conv1_biases = tf.Variable(tf.zeros([32]))
# conv2 权值维度为 64 x 32 x 5 x 5 
  conv2_weights = tf.Variable(
      tf.truncated_normal([5, 5, 32, 64],
                          stddev=0.1,
                          seed=SEED))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
# 全连接层 fc1 权值，神经元数目为512
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal(
          [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
          stddev=0.1,
          seed=SEED))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
# fc2 权值，维度与标签类数目一致
  fc2_weights = tf.Variable(
      tf.truncated_normal([512, NUM_LABELS],
                          stddev=0.1,
                          seed=SEED))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

# 两个网络：训练网络和评测网络
# 它们共享权值

# 实现 LeNet-5 模型，该函数输入为数据，输出为fc2的响应
# 第二个参数区分训练网络还是评测网络
  def model(data, train=False):
"""The Model definition."""
# 二维卷积，使用“不变形”补零（即输出特征图与输入尺寸一致）。
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # 加偏置、过激活函数一块完成
relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # 最大值下采样
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # 第二个卷积层
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
# 特征图变形为2维矩阵，便于送入全连接层
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
# 全连接层，注意“+”运算自动广播偏置
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
# 训练阶段，增加 50% dropout；而评测阶段无需该操作
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  # 训练阶段计算： 对数+交叉熵 损失函数
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, train_labels_node))


  # 全连接层参数进行 L2 正则化
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                  tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
  # 将正则项加入损失函数
  loss += 5e-4 * regularizers

  # 优化器： 设置一个变量，每个批处理递增，控制学习速率衰减
  batch = tf.Variable(0)
  # 指数衰减
  learning_rate = tf.train.exponential_decay(
      0.01,                # 基本学习速率
      batch * BATCH_SIZE,  # 当前批处理在数据全集中的位置
      train_size,          # Decay step.
      0.95,                # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)

  # 用softmax 计算训练批处理的预测概率
  train_prediction = tf.nn.softmax(logits)

  # 用 softmax 计算评测批处理的预测概率
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [optimizer, loss, learning_rate, train_prediction],
          feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
            eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    if FLAGS.self_test:
      print('test_error', test_error)
      assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
          test_error,)
# 程序入口点
if __name__ == '__main__':
  tf.app.run()
```

运行结果：
```
$ python -m tensorflow.models.image.mnist.convolutional
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Extracting data/train-images-idx3-ubyte.gz
Extracting data/train-labels-idx1-ubyte.gz
Extracting data/t10k-images-idx3-ubyte.gz
Extracting data/t10k-labels-idx1-ubyte.gz
Initialized!
Step 0 (epoch 0.00), 2.6 ms
Minibatch loss: 12.054, learning rate: 0.010000
Minibatch error: 90.6%
Validation error: 84.6%
Step 100 (epoch 0.12), 169.5 ms
Minibatch loss: 3.286, learning rate: 0.010000
Minibatch error: 6.2%
Validation error: 7.1%
Step 200 (epoch 0.23), 152.8 ms
Minibatch loss: 3.479, learning rate: 0.010000
Minibatch error: 10.9%
Validation error: 3.8%
Step 300 (epoch 0.35), 153.9 ms
Minibatch loss: 3.213, learning rate: 0.010000
Minibatch error: 6.2%
Validation error: 3.1%
Step 400 (epoch 0.47), 155.2 ms
Minibatch loss: 3.208, learning rate: 0.010000
Minibatch error: 7.8%
Validation error: 2.8%
Step 500 (epoch 0.58), 152.8 ms
Minibatch loss: 3.291, learning rate: 0.010000
Minibatch error: 9.4%
Validation error: 2.6%
Step 600 (epoch 0.70), 152.2 ms
Minibatch loss: 3.202, learning rate: 0.010000
Minibatch error: 6.2%
Validation error: 2.6%
Step 700 (epoch 0.81), 156.4 ms
Minibatch loss: 2.998, learning rate: 0.010000
Minibatch error: 1.6%
Validation error: 2.5%
Step 800 (epoch 0.93), 163.2 ms
Minibatch loss: 3.078, learning rate: 0.010000
Minibatch error: 7.8%
Validation error: 2.0%
Step 900 (epoch 1.05), 152.9 ms
Minibatch loss: 2.927, learning rate: 0.009500
Minibatch error: 3.1%
Validation error: 1.6%
Step 1000 (epoch 1.16), 154.7 ms
Minibatch loss: 2.852, learning rate: 0.009500
Minibatch error: 0.0%
Validation error: 1.7%
Step 1100 (epoch 1.28), 153.5 ms
Minibatch loss: 2.823, learning rate: 0.009500
Minibatch error: 0.0%
Validation error: 1.6%
Step 1200 (epoch 1.40), 150.9 ms
Minibatch loss: 2.913, learning rate: 0.009500
Minibatch error: 7.8%
Validation error: 1.4%
Step 1300 (epoch 1.51), 155.3 ms
Minibatch loss: 2.768, learning rate: 0.009500
Minibatch error: 0.0%
Validation error: 1.7%
Step 1400 (epoch 1.63), 153.1 ms
Minibatch loss: 2.774, learning rate: 0.009500
Minibatch error: 3.1%
Validation error: 1.5%
Step 1500 (epoch 1.75), 151.7 ms
Minibatch loss: 2.880, learning rate: 0.009500
Minibatch error: 6.2%
Validation error: 1.4%
Step 1600 (epoch 1.86), 154.3 ms
Minibatch loss: 2.696, learning rate: 0.009500
Minibatch error: 1.6%
Validation error: 1.4%
Step 1700 (epoch 1.98), 154.1 ms
Minibatch loss: 2.650, learning rate: 0.009500
Minibatch error: 0.0%
Validation error: 1.3%
Step 1800 (epoch 2.09), 150.3 ms
Minibatch loss: 2.666, learning rate: 0.009025
Minibatch error: 1.6%
Validation error: 1.3%
Step 1900 (epoch 2.21), 165.0 ms
Minibatch loss: 2.659, learning rate: 0.009025
Minibatch error: 1.6%
Validation error: 1.2%
Step 2000 (epoch 2.33), 171.5 ms
Minibatch loss: 2.637, learning rate: 0.009025
Minibatch error: 3.1%
Validation error: 1.3%

```
