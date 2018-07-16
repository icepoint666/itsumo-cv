# Alexnet
## Alexnet结构
![ ](./pics/alexnet1.jpg  "alexnet_structure")

## Alexnet简介
alexnet总共包括8层，其中前5层convolutional，后面3层是full-connected，文章里面说的是减少任何一个卷积结果会变得很差，下面我来具体讲讲每一层的构成：

第一层卷积层 输入图像为227\*227\*3(paper上貌似有点问题224\*224\*3)的图像，使用了96个kernels（96,11,11,3），以4个pixel为一个单位来右移或者下移，能够产生5555个卷积后的矩形框值，然后进行response-normalized（其实是Local Response Normalized，后面我会讲下这里）和pooled之后，pool这一层好像caffe里面的alexnet和paper里面不太一样，alexnet里面采样了两个GPU，所以从图上面看第一层卷积层厚度有两部分，池化pool_size=(3,3),滑动步长为2个pixels，得到96个2727个feature。

第二层卷积层使用256个（同样，分布在两个GPU上，每个128kernels（5\*5\*48）），做pad_size(2,2)的处理，以1个pixel为单位移动（感谢网友指出），能够产生27*27个卷积后的矩阵框，做LRN处理，然后pooled，池化以3*3矩形框，2个pixel为步长，得到256个13*13个features。

第三层、第四层都没有LRN和pool，第五层只有pool，其中第三层使用384个kernels（3\*3*\256，pad_size=(1,1),得到256*15*15，kernel_size为（3，3),以1个pixel为步长，得到256\*13\*13）；第四层使用384个kernels（pad_size(1,1)得到256*15*15，核大小为（3，3）步长为1个pixel，得到384\*13\*13）；第五层使用256个kernels（pad_size(1,1)得到384*15*15，kernel_size(3,3)，得到256\*13\*13，pool_size(3，3）步长2个pixels，得到256\*6\*6）。


全连接层： 前两层分别有4096个神经元，最后输出softmax为1000个（ImageNet），注意caffe图中全连接层中有relu、dropout、innerProduct。

（感谢AnaZou指出上面之前的一些问题） paper里面也指出了这张图是在两个GPU下做的，其中和caffe里面的alexnet可能还真有点差异，但这可能不是重点，各位在使用的时候，直接参考caffe中的alexnet的网络结果，每一层都十分详细，基本的结构理解和上面是一致的。
