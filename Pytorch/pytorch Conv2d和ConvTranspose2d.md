# pytorch Conv2d和ConvTranspose2d

### Conv2d
```python
torch.nn.Conv2d(128, 64, (4, 4), stride=2, padding=1)
```
参数含义：

`input_channel_num: 128`

`output_channel_num: 64`

`filter_size: 4 × 4`

`stride: 2 × 2`

`padding: 1`

计算output大小公式：output = ceil( input / stride )

**输入是形如(batchSize, 128, 14, 14)，经过上述Conv2d，输出就是(batchSize, 64, 7, 7)**

我们得到了Conv2d 的输入是 (14, 14) 或者 (15, 15)，输出是 (7, 7)，kernel 是 (4, 4)，stride = 2，那么 padding 就可以计算出来了：

如果输入是取 (14, 14) 的话，(14 - 4 + 2 * padding) / 2 + 1 = 7，此时的 padding 是 1.

如果输入是取 (15, 15) 的话，(15 - 4 + 2 * padding) / 2 + 1 = 7，那么 padding 是 0.5。

### ConvTranspose2d
```python
torch.nn.ConvTranspose2d(128, 64, (4, 4), stride=2, padding=1)
```
参数含义与上面类似

计算output大小公式：output = input * stride

**输入是形如(batchSize, 128, 7, 7)，经过上述Conv2d，输出就是(batchSize, 64, 14, 14)**

