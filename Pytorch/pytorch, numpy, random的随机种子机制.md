# pytorch, numpy, random的随机种子机制

**问题：对于一个项目文件，只在主函数,或在一个地方定义一个随机种子seed，那么调用别的文件的函数，这个seed能不能一直生效呢？**

**回答：是可以的，定义一次，全局生效**

### 验证实验1：主函数定义

主函数文件
```python3
# main.py
import torch
import numpy as np
import random
from test import random_test

seed = 3000
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

random_test() 
```

调用函数所在文件
```python3
# test.py
import torch
import numpy as np
import random

def random_test():
    print(torch.randint(0, 100, [1]))
    print(np.random.randint(0, 100))
    print(random.randint(0, 100))
```

连续多次运行main.py，查看输出结果
```shell
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
```

**说明随机seed只需要在主函数设置一次即可**

### 验证实验2：主函数调用函数定义，主函数调用函数使用
主函数
```python3
# main.py
from util import set_seed
from test import random_test

seed = 3000
set_seed(seed)
random_test() 
```
定义seed的工具文件
```python3
# util.py
import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
```
使用seed产生随机数的函数所在文件
```python3
# test.py
import torch
import numpy as np
import random

def random_test():
    print(torch.randint(0, 100, [1]))
    print(np.random.randint(0, 100))
    print(random.randint(0, 100))
```
同样连续多次运行main.py，查看输出结果
```shell
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
icep@icep-ubuntu1804:~/test$ python3 main.py 
tensor([60])
51
81
```
**说明随机seed只需要在项目可以运行到的地方设置一次即可**
