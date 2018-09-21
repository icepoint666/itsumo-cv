# style transfer实现中用到的一些pytorch用法

### 1.torchvision.transforms
torchvision.transforms是pytorch中的图像预处理包

一般用Compose把多个步骤整合到一起：

比如说

transforms.Compose([

transforms.CenterCrop(10),

transforms.ToTensor(),

])

这样就把两个步骤整合到一起

接下来介绍transforms中的函数

Resize：把给定的图片resize到given size

Normalize：Normalized an tensor image with mean and standard deviation

ToTensor：convert a PIL image to tensor (H*W*C) in range [0,255] to a torch.Tensor(C*H*W) in the range [0.0,1.0]

ToPILImage: convert a tensor to PIL image

Scale：目前已经不用了，推荐用Resize

CenterCrop：在图片的中间区域进行裁剪

RandomCrop：在一个随机的位置进行裁剪

RandomHorizontalFlip：以0.5的概率水平翻转给定的PIL图像

RandomVerticalFlip：以0.5的概率竖直翻转给定的PIL图像

RandomResizedCrop：将PIL图像裁剪成任意大小和纵横比

Grayscale：将图像转换为灰度图像

RandomGrayscale：将图像以一定的概率转换为灰度图像

FiceCrop：把图像裁剪为四个角和一个中心

ColorJitter：随机改变图像的亮度对比度和饱和度

使用的时候,是对单张图片处理：
```python
loader = transforms.Compose([
    transforms.Scale(imsize),
    transforms.ToTensor()])
    
image = Image.open("sample.png")
image = Variable(loader(image))
```
### 2.squeeze与unsqueeze
torch.squeeze() 对于tensor变量进行维度压缩，去除维数为1的的维度。例如一矩阵维度为A*1*B*C*1*D，通过squeeze()返回向量的维度为A*B*C*D。squeeze(a)，表示将a的维数位1的维度删掉，squeeze(a,N)表示，如果第N维维数为1，则压缩去掉，否则a矩阵不变

torch.unsqueeze() 是squeeze()的反向操作，增加一个维度，该维度维数为1，可以指定添加的维度。例如unsqueeze(a,1)表示在1这个维度进行添加

