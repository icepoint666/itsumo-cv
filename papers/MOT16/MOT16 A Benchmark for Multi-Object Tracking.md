# MOT16: A Benchmark for Multi-Object Tracking 
## 简介：
> MOT16是2016年提出的多目标跟踪MOT Challenge系列的一个衡量多目标检测跟踪方法标准的数据集。 
> 官方网站：https://motchallenge.net/ 论文可见：https://arxiv.org/abs/1603.00831 
> MOT16主要标注目标为移动的行人与车辆，是基于MOT15的添加了细化的标注、更多bounding box的数据集， MOT16拥有更加丰富的画面，不同拍摄视角和相机运动，也包含了不同天气状况的视频。是由一批合格的研究人员严格遵从相应的标注准则进行标注的，最后通过双重检测的方法来保证标注信息的高精确度。MOT16标注的运动轨迹为2D。

## 数据集
MOT16数据集共有14个视频序列，其中7个为带有标注信息的训练集，另外7个为测试集。下图第一行为训练集，第二行为测试集。 

![](/pics/mot1.jpg)

下图为MOT16数据集的数据统计表，第一个表为训练集，第二个表为测试集，表格信息包含视频帧率（帧/秒），每帧图像的尺寸，时长，标注box数量，平均每一帧出现的行人数，相机运动情况和拍摄视角以及天气状况。 

![](/pics/mot2.jpg)

MOT16采用了一些较领先的目标检测算法来测试数据集的标注框, 目标检测算法ACF, Fast-RCNN, DPM v5实现

下图显示了用DPM方法检测MOT16数据中目标的统计结果：14个视频序列，表格包含每个视频的目标检测总数（检测出的box），平均每帧目标检测数，检测出的bounding box在画面中的最高、最低的位置。 

![](/pics/mot3.jpg)

MOT16数据集的文档组织格式，所有视频被按帧分为图像，图像统一采用jpeg格式，命名方式为6位数字如：000001.jpg

目标和轨迹信息标注文件为CSV格式，目标信息文件和轨迹信息文件每行都代表一个目标的相关信息，每行都包含9个数值。

### det文件
目标检测文件中内容见下图，第一个值表示目标出现在第几帧，第二个值表示目标运动轨迹的ID号，在目标信息文件中都为-1，第三到第六个值为标注bounding box的坐标尺寸值，第七个值为目标检测表示的confidence score，第八、九个值在目标信息文件中不作标注（都设为-1）。 

![](/pics/mot4.jpg)

### gt文件
下图为目标的轨迹标注文件，第一个值同上，第二个值为目标运动轨迹的ID号，第三个到第六个值的同上，第七个值为目标轨迹是否进入考虑范围内的标志，0表示忽略，1表示active。第八个值为该轨迹对应的目标种类（种类见下面的表格中的label-ID对应情况），第九个值为box的visibility ratio，表示目标运动时被其他目标box包含/覆盖或者目标之间box边缘裁剪情况。 

![](/pics/mot5.jpg)

![](/pics/mot6.jpg)

上面表格中的第12类表示目标检测评价体系考虑到的但是不作为真正例和真反例考虑的类别（原文：which is to be considered by the evaluation script and will neither count as a false negative, nor as a true positive, independent of whether it is correctly recovered or not），第8类表示错检（诱导答案），9-11类表示被遮挡的类别。 
每个（图像序列）视频对应一个‘Sequence-Name.txt’ 包含刚才所有CSV文件的内容。整个数据集为1.9G，训练集中多提供了ground truth.txt作为训练参考。

## 标注规则

1、 Target Class-目标类别划分规则 
MOT16标注的主要是移动中的目标，将所有目标简要分为以下三类： 

Target：(i)移动中的行人与站立的行人； 

Ambiguous：(ii)不处于直立状态的人与人造物（artificial representations） 

Other：(iii)车辆和互相包含/遮挡的目标（vehicles and occluders） 

第一种类别中，由观察者标注所有出现在视野中移动或直立的人，包括在自行车或者滑板上的人，处于弯腰、深蹲、与小孩对话、捡东西状态的行人也同样被考虑在该类别内。 

第二种类别中，包括people-like的目标（模特，出现人的picture，反射的人影），被划分为模糊目标（不同viewer之间的意见变化较大的），不处于直立状态的静态的人（坐着或躺着的）。带着墨镜的人被划分为distractors。 

第三种类别中，标注所有移动的车辆和非机动车（如婴儿车）和其他存在潜在包含/遮挡关系的物体。这个类别中的标注信息仅提供给参赛者训练使用，不算在评价目标检测方法的准则中，静态的车辆或者自行车若没有包含行人则不考虑在内。 

2、 Bounding box alignment 

Bounding box在尽可能紧凑的情况下要包含所标记目标的所有像素点。这意味着一个正在移动的行人的bounding box是长宽不断变化的，如果这个人局部被遮挡，box的尺寸可以参考其他的信息，如影子，反射，上/下一帧的尺寸等。如果一个人正好在图像的边缘部分（被裁剪掉一部分），那么bounding box可以超出该帧图像的大小来标记完整的行人。如果一个物体被部分遮挡或者存在包含问题（e.g.一棵树有很多树枝，如果box把树枝标注进来会过大而把其他无关物体包含进来），那么就用多个box来近似表示该物体。在自行车上的人仅标注该人，不考虑包含他的车，在汽车内的人不做标注。 

3、 Start and end of trajectories 起始与结束时间点 

在标注者确认该物体不属于ambiguous 类别时：Start as early as possible, end as late as possible. 

4、 Minimal size 

虽然有时图像中的行人占很小的尺寸，但是这里要求标注者在人眼可分辨范围内尽可能地标注。（In other words, all targets independent of their size on the image shall be annotated） 

5、 Occlusions遮挡 

主要体现在跟踪标注时，在物体能够被识别无误的情况下尽可能标记，若物体运动时被完全遮挡或者消失，则该物体再次出现时重新设置轨迹ID号。 

6、 Sanity check 检查 

当所有视频被标注完成之后，采用高精度的行人/车辆检测方法来判断标注是否有遗漏、错误，同时人工协助进行审查


