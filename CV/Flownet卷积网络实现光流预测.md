# Flownet: 卷积神经网络实现光流预测

> FlowNet: Learning Optical Flow with Convolutional Networks。 
> 文章已经发布在IEEE International Conference on Computer Vision (ICCV), 2015。 

![ ](../__pics/flownet_1.png)

## Abstract

一般的卷积神经网络都被用来进行分类，最近的一些神经网络结构可以用于对每个像素点进行预测。 

这篇文章主要介绍的就是他们把一般的卷积神经网络去掉全连接层，改成两个网络，一种是比较一般普通的全是卷积层的神经网络，另一个除了卷积层之外还包括一个关联层。并且对这两种网络分别进行点对点的训练，使网络能从一对图片中预测光流场，每秒达到5到10帧率，并且准确率也达到了业界标准。

## 光流原理
是利用图像序列中像素在时间域上的变化以及相邻帧之间的相关性来找到上一帧跟当前帧之间存在的对应关系，从而计算出相邻帧之间物体的运动信息的一种方法。

光流实现的假设前提：
1.相邻帧之间的亮度恒定。

2.相邻视频帧的取帧时间连续，或者，相邻帧之间物体的运动比较“微小”。

3.保持空间一致性；即，同一子图像的像素点具有相同的运动。

因为光流的预测涉及到每个像素点的精确的位置信息，这不仅涉及到图像的特征，还涉及到两个图片之间对应像素点的联系，所以用于光流预测的神经网络与之前的神经网络不同。

## 神经光流网络

![ ](../__pics/flownet_2.png)

他们的两个神经网络大体的思路就是这样。 

首先他们有一个收缩部分，主要由卷积层组成，用于深度的提取两个图片的一些特征。 

但是pooling会使图片的分辨率降低，为了提供一个密集的光流预测，他们增加了一个扩大层，能智能的把光流恢复到高像素。 

他们用back propagation 对这整个网络进行训练。

## 收缩部分网络结构

### FlowNetSimple结构

![ ](../__pics/flownet_3.png)

一个简单的实现方法就是把输入的图片对(两张)叠加在一起，让他们通过一个比较普通的网络结构，

让这个网络来决定如何从这一图片对中提取出光流信息，

这一只有卷积层组成的网络叫做flownetsimple。



这种卷积网络有九个卷积层，

其中的六个stride为2， 每一层后面还有一个非线性的relu操作，

这一网络没有全连接层，所以这个网络不能够把任意大小的图片作为输入，

卷积filter随着卷积的深入递减，第一个7\*7，接下来两个5\*5，之后是3\*3，featuremaps因为stride是2每层递增两倍。

### FlowNetCorr结构
![ ](../__pics/flownet_4.png)

另一个方式 网络先独立的提取俩图片的特征，再在高层次中把这两特征混合在一起。 
这与正常的匹配的方法一致，先提取两个图片的特征，再对这些特征进行匹配，这个网络叫做flownetcorr。

展开看它的关联层

![ ](../__pics/flownet_5.png)

公式如下：

![ ](../__pics/flownet_6.png)

这一公式相当与神经网络的一步卷积层，但普通的卷积是与filter进行卷积，这个是两个数据进行卷积，所以它没有可以训练的权重。

![ ](../__pics/flownet_7.png)

![ ](../__pics/flownet_8.png)

这一公式有ck2的运算， 为了计算速度的原因，我们限制最大的比较位移值。 

![ ](../__pics/flownet_9.png)

![ ](../__pics/flownet_10.png)

![ ](../__pics/flownet_11.png)

## 放大网络结构

![ ](../__pics/flownet_12.png)

扩大部分主要是由上卷基层组成，上卷基层由unpooling（扩大featuremap，与pooling的步骤相反）和一个卷积组成，我们对featuremaps使用upconvolution，并且把它和收缩部分对应的feature map（灰色箭头）以及一个上采样的的光流预测（红色）联系起来。每一步提升两倍的分辨率，重复四次，预测出来的光流的分辨率依然比输入图片的分辨率要小四倍。

这一部的意义就是： 

This way we preserve both the high-level information passed from coarser feature maps and ﬁne local information provided in lower layer feature maps. 

文章中说在这个分辨率时再接着进行双线性上采样的refinement已经没有显著的提高。 

所以采用优化方式：the variational approach。 

记为 +v，这一层需要更高的计算量，但是增加了流畅性，和subpixel-accurate flow filed。

## Flownetsimple与Flownetcorr对比
仅仅看数据会感觉flownetcorr虽然加了关联层，但与s对比并没有太大的改善，因为flownetsimple的正确率也已经很不错了，flownetcorr并没有太大的优势。 
但的是flownetcorr在flyingchair和sintel clean数据集的表现要好于flownetsimple，注意到sintel clean是没有运动blur和fog特效等的，和flyingchair数据集比较类似，这意味着flownetcorr网络能更好的学习训练数据集，更加过拟合over-fitting（文章原话）。 
所以如果使用更好的训练数据集，flownetcorr网络会更有优势。

https://blog.csdn.net/hysteric314/article/details/50529804

https://blog.csdn.net/u013010889/article/details/71189271
