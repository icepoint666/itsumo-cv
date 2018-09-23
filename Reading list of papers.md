# Reading list of papers 
### 2018
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
 
#### 3. 
