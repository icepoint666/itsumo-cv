# LMNN (Large Margin Nearest Neighbor) 大间隔最近邻算法

训练：（特征空间）

首先用普通knn对所有训练节点跑一遍找出每个节点o的三近邻邻居正例（标签和o一致）

然后根据这几个节点的位置，确定一个半径Lmin，在此半径范围内的都会被认为是正例。

有那么一些反例不听话，特征会正好落在该半径范围内，这些节点就是imposter。

假如有个节点能做到半径内没有反例，那么离他最近的imposter定义了一个新的半径Lmax。

引入一个概念Largest margin = Lmax - Lmin

我把这个空间用半定矩阵转化，那么所有的节点坐标都变了。该转化的目的在于降低木变函数数值。通过不断的变化找到一个最小值。由于是凸优化问题，所以是全局最优。

目标函数的意图就在于对任一个节点，把所有的imposter推出去，并且扩大Largest Margin的距离。看出来没，和SVM概念挺像的。

代码：http://www.cs.cornell.edu/~kilian/code/lmnn/lmnn.html

```matlab
% function [L,Det]=lmnnCG(x,y,Kg,'Parameter1',Value1,'Parameter2',Value2,...);
%
% Input:
% x = input matrix (each column is an input vector) 
% y = labels 
% Kg = attract Kg nearest similar labeled vectors 
% 
% Important Parameters:
% outdim = (default: size(x,1)) output dimensionality
% maxiter = maximum number of iterations (default: 1000)
% quiet = true/false surpress output (default=false)  
%
%
% Specific parameters (for experts only):
% initL = define initial matrix L to be passed in (default initL=[])
% thresh = termination criterion (default=1e-05)
% single = true/false performs most computations in sigle precision (default=true)
% GPU = true/false performs computation on GPU (default: false)
%
%
% Output:
%
% L = linear transformation xnew=L*x
%    
% Det.obj = objective function over time
% Det.nimp = number of impostors over time
% Det.pars = all parameters used in run
% Det.time = time needed for computation
% Det.iter = number of iterations
% Det.verify = verify (results of validation - if used) 
```

In particular, most impostor constraints are naturally satisfied and do not need to be enforced during runtime. A particularly well suited solver technique is the working set method, which keeps a small set of constraints that are actively enforced and monitors the remaining (likely satisfied) constraints only occasionally to ensure correctness.

只有小部分的imposter是限制条件，会使距离变化矩阵改变，使得损失函数变化。

注意函数输出的L是转换矩阵，想得到低维样本还要X_new = X \*L

### 代码学习：
```matlab
    %计算每个样本的模长
    sum_X = sum(X .^ 2, 2);
    %不同维度相加操作的函数，DD矩阵具体每个元素：||xi-xj||2
    DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
```
这是计算两个feature矩阵之间距离的方法，比起自己写的for循环，优雅且不容易出错。

举个例子：
```matlab
A= rand(3,1);
B = rand(3,3);
%假设要计算A与B当中每个行向量差的模。也就是sum((B(i,:)-A').^2,2);i = 1:3
sum_A = sum(A.^2);
sum_B = sum(B.^2,2);
DD = sum_A +sum_B - 2*B*A;
%为什么不需要bsxfun因为2016版后不需要这个函数也可以直接扩展维度不同的矩阵，进而加减乘除
```
