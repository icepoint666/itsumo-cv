# Beyond triplet loss: a deep quadruplet network for person re-identiﬁcation论文笔记

本文使用深度学习进行行人检索，侧重点主要在损失函数的改进，提出了 quadruplet loss 用于减小类内方差 和 增加类间方差

** 对于Triplet loss: **
因为用于person-reID, train-set与test-set没有交叉的类，训练集中的人与测试集的人不一样，所以训练集的generalization capability很弱对于测试集。

内在原因(underlying reason) 是因为训练模型造成相对大的intra-class 变动。

所以像下图，减少intra-class 变动幅度，扩大inter-class 变动幅度可以减少训练模型的泛化错误

![](pics/quad.jpg)

Our designed loss simultaneously considers the following two aspects in one quadruplet:

1) obtaining correct orders for pairs w.r.t the same probe image (e.g. B1B3 < B1A3 in Fig. 1); 

2) pushing away negative pairs from positive pairs w.r.t different probe images (e.g. C1C2 < B1A3 in Fig. 1).

第一个方面和triplet loss的想法一致，

第二个方面关注点更多集中在减少intra-class variations，扩大inter-class variations

关于这两个方面的平衡用两个常量margin控制。

第二个方面对训练集测试分数没有太大影响，但是对测试集比较有好处，增强模型的泛化能力。

https://blog.csdn.net/zhangjunhit/article/details/70239948
