# numpy.random.choice()
## 概述：
可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。

## 说明：
```
numpy.random.choice(a, size=None, replace=True, p=None)
a : 1-D array-like or int
    If an ndarray, a random sample is generated from its elements. 
    If an int, the random sample is generated as if a was np.arange(n)

size : int or tuple of ints, optional 

replace : boolean, optional
    Whether the sample is with or without replacement

p : 1-D array-like, optional
    The probabilities associated with each entry in a. If not given the sample assumes a uniform distribution over all entries in a.
```
## 示例
```
>>> np.random.choice(5, 3)
array([0, 3, 4])

>>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
array([3, 3, 0])

>>> np.random.choice(5, 3, replace=False)
array([3,1,0])

>>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
array([2, 3, 0])

>>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']

>>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],

```
https://blog.csdn.net/autoliuweijie/article/details/51982514
