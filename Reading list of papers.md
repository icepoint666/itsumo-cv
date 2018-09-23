# Reading list of papers 
### **2018**
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


**===========把事情做好的标志就是让别人可以轻松明白===========**
