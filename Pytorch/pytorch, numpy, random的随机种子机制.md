# pytorch, numpy, random的随机种子机制

**问题：对于一个项目文件，只在主函数定义一个随机种子seed，那么调用别的文件的函数，这个seed能不能一直生效呢？**

**回答：是可以的**

验证实验：

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

