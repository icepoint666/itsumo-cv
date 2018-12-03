# Reading list of papers 
### **2018**✈
##### 1. Active Shape Models --Their Training and Application. T.F.Cootes et al. CVIU 1995.
`2018.4.25`
`Face Recognition`

ASM算法，基于点分布模型，标定67个点的人脸训练图像，对每个特征点构建局部特征，用model来翻译图像，提取特征(shape, position),找寻best parameters, optimising a cost function

##### 2. Robust Real-Time Face Detection. P.Viola et al. IJCV 2004.
`2018.6.23`
`Face Recognition`

创新点：

 - 1). Integral Image 积分图像：用integral image来计算rectangle features，加速haar特征计算的一个巧妙点子，去掉特征计算中的冗余，思想类似于前缀和
 - 2). AdaBoost，从很多潜在的features中，选择小数量的重要features，对AdaBoost的思想进行改造，一个haar特征对应一个弱分类器，弱特征组合成强特征，弱分类器组合成强分类器（特征选择+分类器融合）
 - 3). Cascade Structure 级联结构：用于提升速度，又粗到精的检测策略，加速的同时可以保证精度，决定期望有人脸的区域
 
##### 3. FaceNet: A Unified Embedding for Face Recognition and Clustering. F.Schroff et al. CVPR 2015.
`2018.8.10`
`Face Recognition`
`Metric Learning`

谷歌人脸检测算法，发表于 CVPR 2015，利用相同人脸在不同角度等姿态的照片下有高内聚性，不同人脸有低耦合性，提出使用 cnn + triplet mining 方法，在 LFW 数据集上准确度达到 99.63%

创新点：提出三元损失函数，之前二元损失函数的目标是把相同个体的人脸特征映射到空间中的相同点，而三元损失函数目标是映射到相同的区域，使得类内距离小于类间距离。

##### 4. Beyond triplet loss: a deep quadruplet network for person re-identification. W.Chen et al. CVPR 2017.
`2018.8.11`
`Person ReID`
`Metric Learning`

创新点：提出四元损失函数，在保证分类能力的情况下提升泛化能力，相当于由只度量类与类之间的相对距离，增加了度量绝对距离的项，泛化性能自然提升，对于训练类集和测试类集不同的问题领域，会有更好的效果，face, person-reid。

##### 5. Multi-Object Tracking with Quadruplet Convolutional Neural Networks. J.Son et al. CVPR 2017.
`2018.8.1`
`MOT`
`Loss Function`

提出一种quadruplet CNN，通过使用quadruplet losses来associate object detections across frames

创新点：
 - 1). 通过设计一种新的quadruplet loss来度量frames之间detections是否属于同一个物体
 - 2). 使用multi-task loss并设计一种 end-to-end 神经网路， jointly learn bounding box regression and object association
 - 3). Data association使用Minimax Label Propagation算法
 
##### 6. Image Style Transfer Using Convolutional Neural Network. Gatys at al. CVPR 2016.
`2018.9.18`
`Style Transfer`
`Texture Synthesis`

之前的style transfer只能实现比较专一的某种特定对象下的风格迁移，例如手写数组，人脸等，这个方法更加general

Texture Synthesis Using Convolutional Neural Networks的升华

风格转移，输出的图片C = A(content) + B(style)，就是将图片B的风格转移到图片C中，同时保留图片A的内容。这看起来像是PS就能实现的功能，直接将两个图层叠加融合即可。但其实不然，保留图片A的内容是所有CNN网络都可以实现的，但style是更为抽象的特征。

核心思想：
 - 1). Gram Matrix提取style features
 - 2). multi-task losses 同时考虑high level content，与每一层的style
 - 3). 参数alpha, beta, weight的选取，从而适应general的图像情况

##### 7. Perceptual Losses for Real-Time Style Transfer and Super-Resolution. Jcjohns et al. ECCV 2016.
`2018.9.22`
`Style Transfer`

[[project](https://github.com/hzy46/fast-neural-style-tensorflow)]

风格转移的两种方法:

1). 基于图片迭代的描述性神经网络

这一方法会从随机噪声开始，通过反向传播迭代更新（尚未知晓的）风格化图像。图像迭代的目标是最小化总损失，这样风格化后的图像就能同时将内容图像的内容与风格图像的风格匹配起来。

需要指定输入图片与风格图片，然后进行足够多的迭代才能有比较好的效果，每一次运行都是重新训练一个模型，且不可复用。

2). 基于模型迭代的生成式神经网络

这种方法更像是为了解决“基于图片迭代的描述性神经网络”在风格转移中效率过低而存在的，也被成为“快速”神经风格迁移，主要解决了前者的计算速度和计算成本问题。

核心是先训练保存一些用户常用的“风格”参数。用户输入需要转换风格的图片A到网络中，再指定网络调用B风格的参数，输出的图片C其实只是网络最后一层的输出，中间不会涉及过多的参数调整及优化。

与第一种方法对比，基于模型迭代的生成式神经网络更像是一个网络的“test”部分，其只是输出结果，做部分参数的调整，但优劣性不予保证；基于图片迭代的描述性神经网络就是一个网络的“train”部分，其需要进行多次的权重调整及优化，才能使得初始化的噪声图片有比较好的输出。

本文real-time就是因为实现的是第二种方法

##### 9. Why do deep convolutional networks generalize so poorly to small image transformations? Azulay et al. CVPR 2018.
`2018.10.25`
`CNN`

通常认为CNN对图像平移具有不变性，但发现当图像在当前平面上平移几个像素后，现代CNN(VGG16, Resnet50, InceptionResnetv2)输出会发生巨大变化。
对模型输入图像做了微小的平移，放大，形变后，会导致模型评分出现“滑铁卢”。

产生这个现象的主因：

1). 现代CNN体系结构没有遵循经典采样定理，根据nyquist theorem，从信号学角度分析，CNN小规模平移无法保证平移不变性，主要是subsampling在平移不变性上的失败。(笔记里实验证明，待进一步理解）

2). 常用图像数据集(eg: Imagenet),由于数据普遍中心就在图像中心，存在统计偏差，所以CNN泛化能力其实还不够好，CNN成功的关键在于归纳偏差的方法。





**===============把事情做好的标志就是让别人可以轻松明白===============**
