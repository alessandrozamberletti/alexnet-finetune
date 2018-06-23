# caffe-tensorflow-alexnet
```python
dataset = Dataset('res/data', AlexNet.scale_size, mean_image=AlexNet.mean_image)
images, labels = dataset.load()
X_train, X_val, y_train, y_val = train_val_split(images, labels)

alex_net = AlexNet(dataset.num_classes, 'res/alexnet_caffemodel.npy')
alex_net.fit(X_train, X_val, y_train, y_val, freeze=True, epochs=1000, lr=0.001)
```

## Instructions
Use the following dataset structure: ```res/data/class_{0,..,K}/image_{0,..,N}.jpg```\
Run:\
```console
az@ubuntu:~$ python fine_tune.py
```
If the converted AlexNet caffemodel (source: [BVLC](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet))
is not in ```res/``` folder, it will be downloaded from
[here](https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet_caffemodel.npy?dl=1).

## Dependencies
```console
az@ubuntu:~$ python -V
Python 2.7.10
```

```console
az@ubuntu:~$ python -c 'import tensorflow as tf; print(tf.__version__)'
1.8.0
```

## Resources
Original caffemodel and prototxt: [https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet]\
Caffe-tensorflow conversion project: [https://github.com/ethereon/caffe-tensorflow]\
Evaluation dataset: [https://www.kaggle.com/c/dogs-vs-cats]
