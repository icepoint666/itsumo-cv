# imwrite保存图片
```
bool imwrite(const string& filename, InputArray img, const vector<int>& params=vector<int>() )
```
该函数是把程序中的Mat类型的矩阵保存为图像到指定位置。

参数filename为所需保存图像的文件目录和文件名。这里的文件名需要带有图像格式后缀的，目前OpenCV该函数只支持JPEG,PNG,PPM,PGM,PBM，TIFF等。并不是所有Mat类型都支持。

不支持.jpg
