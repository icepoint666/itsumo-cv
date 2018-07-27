# Triplet Loss

> Triplet Loss是深度学习中的一种损失函数，用于训练差异性较小的样本，如人脸等
> Feed数据包括锚（Anchor）示例、正（Positive）示例、负（Negative）示例，
> 通过优化锚示例与正示例的距离小于锚示例与负示例的距离，实现样本的相似性计算。

Triplet Loss的核心是锚示例、正示例、负示例共享模型，通过模型，将锚示例与正示例聚类，远离负示例。

- 输入：三个输入，即锚示例、正示例、负示例，不同示例的结构相同；
- 模型：一个共享模型，支持替换为任意网络结构；
- 输出：一个输出，即三个模型输出的拼接。

Triplet Loss损失函数的计算公式如下：

<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda \theta_k^{(j)}" />

