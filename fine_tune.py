import os.path
import wget

from sklearn.model_selection import train_test_split
from dataset import Dataset
from alexnet import AlexNet

dataset = Dataset('res/data', AlexNet.scale_size, mean_image=AlexNet.mean_image)
images, labels = dataset.load()

X_train, X_test, y_train, y_test = train_test_split(images, labels)

weights_file = 'res/alexnet_caffemodel.npy'
if not os.path.exists(weights_file):
    wget.download('https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet_caffemodel.npy?dl=1', 'res')

alex_net = AlexNet(dataset.num_classes, weights_file)
alex_net.fit(X_train, X_test, y_train, y_test)
