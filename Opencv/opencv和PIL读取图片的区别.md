# opencv和PIL读取图片的区别
### opencv读取图片:
读取的是**bgr空间**
```python
import cv2
img = cv2.imread("test.png")
```
转成PIL读取的格式,**rgb空间**
```python
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
```
### PIL(pillow)读取图片:
读取的是**rgb空间**
```python
from PIL import Image
image = Image.open("test.png")
```
转成opencv读取的格式,**bgr空间**
```python
img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
```
