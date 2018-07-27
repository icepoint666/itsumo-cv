# caffe中的train_val.prototxt
```prototxt
train_val.prototxt训练网络配置文件

    name: "AlexNet"                      #框架的名字
    layer {
        name: "data"                        #该层的名字
        type: "Data"                         #该层的类型
        top: "data"                           #top/bottom botton用来输入数据，top用来输出数据
        top: "label"                          #多个top/botto表示多个数据的输入或输出
        include {                              #具有include参数，该层会表示是训练或者测试，如果没有表示该层即在测试又在训练模型中
            phase: TRAIN                  #表示是训练
         }
        transform_param {
            mirror: true                     #true表示开启镜像，false表示关闭镜像
            crop_size: 227                 #剪裁一个227*227的图块
            mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"   #训练的均值文件
        }
        data_param {
            source: "examples/imagenet/ilsvrc12_train_lmdb"        #训练的图片转换成的lmdb数据格式
            batch_size: 256                        #每次处理数据的个数
            backend: LMDB                       #数据的格式
       }
    }
    layer {
        name: "data"                                          
        type: "Data"
        top: "data"                         #如果有两个top一般一个是data一个是label
        top: "label"
        include {
            phase: TEST                   #表示测试
         }
        transform_param {
            mirror: false                   #测试不生成镜像
            crop_size: 227                #裁剪的图片大小
            mean_file: "data/ilsvrc12/imagenet_mean.binaryproto"   #测试的均值文件
        }
        data_param {
            source: "examples/imagenet/ilsvrc12_val_lmdb"          #测试的图片转换成的lmdb数据格式
            batch_size: 50           #每次测试的图片个数 如果显卡或内存不是很强 可以将每次测试的图片数减小，测试的次数增多，测试的次数在solver.prototxt中
            backend: LMDB          #数据的格式
         }
    }
    layer {
        name: "conv1"                   #第一卷积层
        type: "Convolution"           #卷积
        bottom: "data"                  #输入的数据是data层
        top: "conv1"                      #输出是conv1
        param {
            lr_mult: 1                       #权值的学习率系数，学习率是solver.prototxt中的base_lr * lr_mult
            decay_mult: 1                #权值的衰减
        }
        param {
            lr_mult: 2                       #偏置的学习率系数
            decay_mult: 0	#权值的衰减
        }
        convolution_param {        #卷积的参数
            num_output: 96           #卷积核的个数
            kernel_size: 11              #卷积核大小 11*11  如果卷积核的长和宽不等，需要用 kernel_h 和 kernel_w 分别设定
            stride: 4                         #卷积核的步长
            weight_filler {                #权值的初始化
                type: "gaussian"        #很多时候用xavier算法来初始化，也可以用Gaussian 高斯分布来初始化
                std: 0.01                     #标准差
            }
            bias_filler {                      #偏置的初始化
                type: "constant"          #默认constant固定值
                value: 0                        #固定值多少
            }
      }
}
layer {
  name: "relu1"                            #激活层
  type: "ReLU"                              #激活函数relu
  bottom: "conv1"                        #输入是conv1
  top: "conv1"                               #输出是conv1
}
layer {
  name: "norm1"                          #对输入的局部区域进行归一化，达到侧抑制的效果
  type: "LRN"
  bottom: "conv1"                        #输入conv1
  top: "norm1"                              #输出norml
  lrn_param {
    local_size: 5         #默认5 如果是跨通道LRN，则表示求和的通道数；如果是在通道内LRN，则表示求和的正方形区域长度。 
    alpha: 0.0001                            #默认1 归一化公式中的参数
    beta: 0.75                                  #归一化公式中的参数
  }
}
layer {                                                 
    name: "pool1"                   #第一池化层pool1
    type: "Pooling"                  #池化操作
    bottom: "norm1"               #该层的输入是norml
    top: "pool1"                       #该层的输出是pool1
    pooling_param {                #池化层参数
        pool: MAX                     #池化方法 默认是MAX即池化核内的最大值。目前可用的方法有MAX, AVE, 或STOCHASTIC
        kernel_size: 3                 #池化核的大小
        stride: 2                          #池化的步长
#pad: 2                           #池化的边缘扩充和卷积层一样
      }
}
layer {
    name: "conv2"                   #第二卷积层
    type: "Convolution"
    bottom: "pool1"
    top: "conv2"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    convolution_param {
        num_output: 256                       #卷积核的个数变成256
        pad: 2                                         #边沿扩充是2 也就是长宽都增加4个像素点
        kernel_size: 5                              #卷积核5*5
        group: 2                                      #分组 默认为1 卷积的分组可以减少网络的参数
        weight_filler {                              #权值参数设置
            type: "gaussian"                      #高斯分布
            std: 0.01                                   #标准差是0.01
        }
        bias_filler {                                    #偏置参数设置
            type: "constant"                        #固定值
            value: 0.1                                   #值0.1
       }
   }
}
layer {
    name: "relu2"                                    #激活层
    type: "ReLU"
    bottom: "conv2"
    top: "conv2"
}
layer {
    name: "norm2"                                 #归一化
    type: "LRN"
    bottom: "conv2"
    top: "norm2"
    lrn_param {
        local_size: 5
        alpha: 0.0001
        beta: 0.75
    }
}
layer {                                                 
    name: "pool2"                                 #第二池化层
    type: "Pooling"
    bottom: "norm2"
    top: "pool2"
    pooling_param {
        pool: MAX
        kernel_size: 3                                #卷积核大小 3*3
        stride: 2                                         #步长 2
    }
}
layer {
    name: "conv3"
    type: "Convolution"
    bottom: "pool2"
    top: "conv3"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    convolution_param {
        num_output: 384
        pad: 1
        kernel_size: 3
        weight_filler {
            type: "gaussian"
            std: 0.01
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
}
layer {
    name: "relu3"
    type: "ReLU"
    bottom: "conv3"
    top: "conv3"*
}
layer {
    name: "conv4"
    type: "Convolution"
    bottom: "conv3"
    top: "conv4"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
   convolution_param {
       num_output: 384
       pad: 1
       kernel_size: 3
       group: 2
       weight_filler {
           type: "gaussian"
           std: 0.01
       }
       bias_filler {
           type: "constant"
           value: 0.1
       }
    }
}
layer {
    name: "relu4"
    type: "ReLU"
    bottom: "conv4"
    top: "conv4"
}
layer {
    name: "conv5"
    type: "Convolution"
    bottom: "conv4"
    top: "conv5"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    convolution_param {
        num_output: 256
        pad: 1
        kernel_size: 3
        group: 2
        weight_filler {
            type: "gaussian"
            std: 0.01
         }
         bias_filler {
             type: "constant"
            value: 0.1
        }
    }
}
layer {
    name: "relu5"
    type: "ReLU"
    bottom: "conv5"
    top: "conv5"
}
layer {
    name: "pool5"
    type: "Pooling"
    bottom: "conv5"
    top: "pool5"
    pooling_param {
        pool: MAX
        kernel_size: 3
        stride: 2
    }
}
layer {
    name: "fc6"                              #全连接层
    type: "InnerProduct"              #全连接层实际上也是一种卷积层，只是它的卷积核大小和原数据大小一致。因此它的参数基本和卷积层的参数一样
    bottom: "pool5"                      #输入是pool5
    top: "fc6"                                 #输出是fc6
    param { 
        lr_mult: 1                             #权值学习率系数
        decay_mult: 1                      #权值衰减
    }
    param {
        lr_mult: 2                             #偏置的学习率系数
        decay_mult: 0                      #偏置衰减
    }
    inner_product_param {
        num_output: 4096               #输出的个数
        weight_filler {
            type: "gaussian"              #高斯分布
            std: 0.005                         #标准差
        }
        bias_filler { 
            type: "constant"
            value: 0.1                         #偏置固定值0.1
        }
    }
}
layer {
    name: "relu6"
    type: "ReLU"
    bottom: "fc6"
    top: "fc6"
}
layer {
    name: "drop6"                             #对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
    type: "Dropout"                           #防止过拟合
    bottom: "fc6"                               #该层输入fc6
    top: "fc6"                                     #该层输出fc6
    dropout_param {
        dropout_ratio: 0.5                    #dropout 的概率
    }
}
layer {
    name: "fc7"
    type: "InnerProduct"
    bottom: "fc6"
    top: "fc7"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    inner_product_param {
        num_output: 4096
        weight_filler {
            type: "gaussian"
            std: 0.005
        }
        bias_filler {
            type: "constant"
            value: 0.1
        }
    }
}
layer {
    name: "relu7"
    type: "ReLU"
    bottom: "fc7"
    top: "fc7"
}
layer {
    name: "drop7"
    type: "Dropout"
    bottom: "fc7"
    top: "fc7"
    dropout_param {
        dropout_ratio: 0.5
    }
}
layer {
    name: "fc8"
    type: "InnerProduct"
    bottom: "fc7"
    top: "fc8"
    param {
        lr_mult: 1
        decay_mult: 1
    }
    param {
        lr_mult: 2
        decay_mult: 0
    }
    inner_product_param {
        num_output: 1000
        weight_filler {
            type: "gaussian"
            std: 0.01
        }
        bias_filler {
            type: "constant"
            value: 0
        }
    }
}
layer {             #caffe中计算Accuracy时，是通过比较最后一个全连接层（神经元个数=类别数、但没有加入activation function）的输出和数据集的labels来得到的，计算过程在AccuracyLayer中实现
    name: "accuracy"                    #正确率层
    type: "Accuracy"                     #利用fc8的输出得到数据集的预测labels
    bottom: "fc8"	#最后一层全连接作为输入
    bottom: "label"                       #数据层的lable作为另一个输入
    top: "accuracy"                       #输出正确率
    include {
        phase: TEST                         #测试的正确率
    }
}
layer {
    name: "loss"                            #丢失率
    type: "SoftmaxWithLoss"
    bottom: "fc8"
    bottom: "label"
    top: "loss"

}
```
