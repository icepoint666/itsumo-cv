# torchvision.transform
> PyTorch框架中有一个非常重要且好用的包：torchvision，该包主要由3个子包组成，分别是：torchvision.datasets、torchvision.models、torchvision.transforms。

这篇博客介绍torchvision.transforms。torchvision.transforms这个包中包含resize、crop等常见的data augmentation操作，基本上PyTorch中的data augmentation操作都可以通过该接口实现。该包主要包含两个脚本：transformas.py和functional.py，前者定义了各种data augmentation的类，在每个类中通过调用functional.py中对应的函数完成data augmentation操作。

```python
import torchvision
import torch
train_augmentation = torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                    torchvision.transforms.RandomCrop(224),                                                                            
                                                    torchvision.transofrms.RandomHorizontalFlip(),
                                                    torchvision.transforms.ToTensor(),
                                                    torch vision.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])
                                                    ])

Class custom_dataread(torch.utils.data.Dataset):
    def __init__():
        ...
    def __getitem__():
        # use self.transform for input image
    def __len__():
        ...

train_loader = torch.utils.data.DataLoader(
    custom_dataread(transform=train_augmentation),
    batch_size = batch_size, shuffle = True,
    num_workers = workers, pin_memory = True)
```

这里定义了resize、crop、normalize等数据预处理操作，并最终作为数据读取类custom_dataread的一个参数传入，可以在内部方法__getitem__中实现数据增强操作。

主要代码在transformas.py脚本中，这里仅介绍常见的data augmentation操作，源码如下： 

首先是导入必须的模型，这里比较重要的是from . import functional as F，也就是导入了functional.py脚本中具体的data augmentation函数。\_\_all\_\_列表定义了可以从外部import的函数名或类名。
```python
from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

from . import functional as F

__all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize",
"Scale", "CenterCrop", "Pad", "Lambda", "RandomCrop", 
"RandomHorizontalFlip", "RandomVerticalFlip", "RandomResizedCrop", 
"RandomSizedCrop", "FiveCrop", "TenCrop","LinearTransformation", 
"ColorJitter", "RandomRotation", "Grayscale", "RandomGrayscale"]
```
Compose这个类是用来管理各个transform的，可以看到主要的__call__方法就是对输入图像img循环所有的transform操作
```python
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
```
ToTensor类是实现：Convert a PIL Image or numpy.ndarray to tensor 的过程，在PyTorch中常用PIL库来读取图像数据，因此这个方法相当于搭建了PIL Image和Tensor的桥梁。另外要强调的是在做数据归一化之前必须要把PIL Image转成Tensor，而其他resize或crop操作则不需要。
```python
class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
```
ToPILImage顾名思义是从Tensor到PIL Image的过程，和前面ToTensor类的相反的操作。
```python
class ToPILImage(object):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
            If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
            1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
            2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
            3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
            ``int``, ``float``, ``short``).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    """
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, pic):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        return F.to_pil_image(pic, self.mode)

    def __repr__(self):
        return self.__class__.__name__ + '({0})'.format(self.mode)
```
Normalize类是做数据归一化的，一般都会对输入数据做这样的操作，公式也在注释中给出了，比较容易理解。前面提到在调用Normalize的时候，输入得是Tensor，这个从__call__方法的输入也可以看出来了。
```python
class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
```
Resize类是对PIL Image做resize操作的，几乎都要用到。这里输入可以是int，此时表示将输入图像的短边resize到这个int数，长边则根据对应比例调整，图像的长宽比不变。如果输入是个(h,w)的序列，h和w都是int，则直接将输入图像resize到这个(h,w)尺寸，相当于force resize，所以一般最后图像的长宽比会变化，也就是图像内容被拉长或缩短。注意，在__call__方法中调用了functional.py脚本中的resize函数来完成resize操作，因为输入是PIL Image，所以resize函数基本是在调用Image的各种方法。如果输入是Tensor，则对应函数基本是在调用Tensor的各种方法，这就是functional.py中的主要内容。
```python
class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
CenterCrop是以输入图的中心点为中心点做指定size的crop操作，一般数据增强不会采用这个，因为当size固定的时候，在相同输入图像的情况下，N次CenterCrop的结果都是一样的。注释里面说明了size为int和序列时候尺寸的定义。
```python
class CenterCrop(object):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```

相比前面的CenterCrop，这个RandomCrop更常用，差别就在于crop时的中心点坐标是随机的，并不是输入图像的中心点坐标，因此基本上每次crop生成的图像都是有差异的。就是通过 i = random.randint(0, h - th)和 j = random.randint(0, w - tw)两行生成一个随机中心点的横纵坐标。注意到在__call__中最后是调用了F.crop(img, i, j, h, w)来完成crop操作，其实前面CenterCrop中虽然是调用 F.center_crop(img, self.size)，但是在F.center_crop()函数中只是先计算了中心点坐标，最后还是调用F.crop(img, i, j, h, w)完成crop操作。

```python
class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = F.pad(img, self.padding)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
RandomHorizontalFlip类也是比较常用的，是随机的图像水平翻转，通俗讲就是图像的左右对调。从该类中的__call__方法可以看出水平翻转的概率是0.5。
```python
class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
```
同样的，RandomVerticalFlip类是随机的图像竖直翻转，通俗讲就是图像的上下对调。
```python
class RandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return F.vflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
```
RandomResizedCrop类也是比较常用的，个人非常喜欢用。前面不管是CenterCrop还是RandomCrop，在crop的时候其尺寸是固定的，而这个类则是random size的crop。该类主要用到3个参数：size、scale和ratio，总的来讲就是先做crop（用到scale和ratio），再resize到指定尺寸（用到size）。做crop的时候，其中心点坐标和长宽是由get_params方法得到的，在get_params方法中主要用到两个参数：scale和ratio，首先在scale限定的数值范围内随机生成一个数，用这个数乘以输入图像的面积作为crop后图像的面积；然后在ratio限定的数值范围内随机生成一个数，表示长宽的比值，根据这两个值就可以得到crop图像的长宽了。至于crop图像的中心点坐标，也是类似RandomCrop类一样是随机生成的。
```python
class RandomResizedCrop(object):
    """Crop the given PIL Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback
        w = min(img.size[0], img.size[1])
        i = (img.size[1] - w) // 2
        j = (img.size[0] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly cropped and resize image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
FiveCrop类，顾名思义就是从一张输入图像中crop出5张指定size的图像，这5张图像包括4个角的图像和一个center crop的图像。曾在TSN算法的看到过这种用法。
```python
class FiveCrop(object):
    """Crop the given PIL Image into four corners and the central crop

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size

    def __call__(self, img):
        return F.five_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
TenCrop类和前面FiveCrop类类似，只不过在FiveCrop的基础上，再将输入图像进行水平或竖直翻转，然后再进行FiveCrop操作，这样一张输入图像就能得到10张crop结果。
```python
class TenCrop(object):
    """Crop the given PIL Image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default)

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        vertical_flip(bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops
    """

    def __init__(self, size, vertical_flip=False):
        self.size = size
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
            self.size = size
        self.vertical_flip = vertical_flip

    def __call__(self, img):
        return F.ten_crop(img, self.size, self.vertical_flip)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)
```
LinearTransformation类是用一个变换矩阵去乘输入图像得到输出结果。
```python
class LinearTransformation(object):
    """Transform a tensor image with a square transformation matrix computed
    offline.

    Given transformation_matrix, will flatten the torch.*Tensor, compute the dot
    product with the transformation matrix and reshape the tensor to its
    original shape.

    Applications:
    - whitening: zero-center the data, compute the data covariance matrix
                 [D x D] with np.dot(X.T, X), perform SVD on this matrix and
                 pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
    """

    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be whitened.

        Returns:
            Tensor: Transformed image.
        """
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1)
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += (str(self.transformation_matrix.numpy().tolist()) + ')')
        return format_string
```
ColorJitter类也比较常用，主要是修改输入图像的4大参数值：brightness, contrast and saturation，hue，也就是亮度，对比度，饱和度和色度。可以根据注释来合理设置这4个参数。
```python
class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        np.random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'
```
RandomRotation类是随机旋转输入图像，也比较常用，具体参数可以看注释，在F.rotate()中主要是调用PIL Image的rotate方法。
```python
class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = np.random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        return self.__class__.__name__ + '(degrees={0})'.format(self.degrees)
```
Grayscale类是用来将输入图像转成灰度图的，这里根据参数num_output_channels的不同有两种转换方式。
```python
class Grayscale(object):
    """Convert image to grayscale.

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
        - If num_output_channels == 1 : returned image is single channel
        - If num_output_channels == 3 : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        return F.to_grayscale(img, num_output_channels=self.num_output_channels)

    def __repr__(self):
        return self.__class__.__name__ + '()'
```
RandomGrayscale类和前面的Grayscale类类似，只不过变成了按照指定的概率进行转换。
```python
class RandomGrayscale(object):
    """Randomly convert image to grayscale with a probability of p (default 0.1).

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """

    def __init__(self, p=0.1):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        num_output_channels = 1 if img.mode == 'L' else 3
        if random.random() < self.p:
            return F.to_grayscale(img, num_output_channels=num_output_channels)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
```

https://blog.csdn.net/yt4766269/article/details/77923422
