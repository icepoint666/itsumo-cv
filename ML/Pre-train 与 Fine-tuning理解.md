# Pre-train 与 Fine-tuning

Pre-train的model:

就是指之前被训练好的Model, 比如很大很耗时间的model, 你又不想从头training一遍。这时候可以直接download别人训练好的model， 里面保存的都是每一层的parameter配置情况。(Caffe里对于ImageNet的一个model, 我记得是200+M的model大小)。你有了这样的model之后，可以直接拿来做testing, 前提是你的output的类别是一样的。

如果不一样咋办，但是恰巧你又有一小部分的图片可以留着做fine-tuning, 一般的做法是修改最后一层softmax层的output数量，比如从Imagenet的1000类，降到只有20个类，那么自然最后的InnerProducet层，你需要重新训练，然后再经过Softmax层，再训练的时候，可以把除了最后一层之外的所有层的learning rate设置成为0， 这样在traing过程，他们的parameter 就不会变，而把最后一层的learning rate 调的大一点，让他尽快收敛，也就是Training Error尽快等于0. 

https://blog.csdn.net/wangzuhui0430/article/details/48156717
