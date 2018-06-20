# caffe-tensorflow-alexnet
```python
data = Dataset('res/data', AlexNet.scale_size, mean_image=AlexNet.mean_image)
images, labels = data.load()
X_train, X_test, y_train, y_test = train_test_split(images, labels)

alex_net = AlexNet(data.num_classes, 'res/alexnet_caffemodel.npy')
alex_net.fit(X_train, X_test, y_train, y_test, epochs=100, augment_data=True)
```

## Resources
original caffemodel and prototxt: [https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet]

caffemodel-tensorflow conversion project: [https://github.com/ethereon/caffe-tensorflow]

evaluation dataset: [https://www.kaggle.com/c/dogs-vs-cats]
