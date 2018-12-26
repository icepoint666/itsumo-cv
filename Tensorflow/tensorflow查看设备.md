# Tensorflow查看设备
```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```
```
[name: "/device:CPU:0"
device_type: "CPU"
memory_limit: 268435456
locality {
}
incarnation: 16230250535394445197
, name: "/device:GPU:0"
device_type: "GPU"
memory_limit: 7411086132
locality {
  bus_id: 1
  links {
  }
}
incarnation: 6187650455324439640
physical_device_desc: "device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1"
]

```
https://blog.csdn.net/sinat_29957455/article/details/80636683
