# opencv常量问题
In OpenCV 2.x, it appears that the constants are of the form cv2.cv.CV_CAP_X,
while in OpenCV 3.x, they are cv2.CAP_X
示例报错
```
total = int((vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)/100)*video_perc)
AttributeError: 'module' object has no attribute 'cv'
```
