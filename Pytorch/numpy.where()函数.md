### numpy.where(condition[, x, y])
1、这里x,y是可选参数，condition是条件，这三个输入参数都是array_like的形式；而且三者的维度相同

2、当conditon的某个位置的为true时，输出x的对应位置的元素，否则选择y对应位置的元素；

3、如果只有参数condition，则函数返回为true的元素的坐标位置信息；
