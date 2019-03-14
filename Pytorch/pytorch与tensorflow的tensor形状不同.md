### pytorch tensor:
(b, c, h, w): (batch_size, num_channel, height, width)


### tensorflow tensor:
(b, h, w, c): (batch_size, height, width, num_channel)

需要注意: 有时候tensorflow进行运行卷积函数的时候，可能是stride=(1, 2, 2, 1)，就是因为步长不同。
