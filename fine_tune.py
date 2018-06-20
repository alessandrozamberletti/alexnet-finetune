import os.path
import wget

from sklearn.model_selection import train_test_split
from dataset import Dataset
from alexnet import AlexNet

dataset = Dataset('res/data', AlexNet.SCALE_SIZE, mean_image=AlexNet.MEAN_IMAGE)
images, labels = dataset.load()

X_train, X_val, y_train, y_val = train_test_split(images, labels)

weights_file = 'res/alexnet_caffemodel.npy'
if not os.path.exists(weights_file):
    wget.download('https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet_caffemodel.npy?dl=1', 'res')

alex_net = AlexNet(dataset.num_classes, weights_file)
alex_net.fit(X_train, X_val, y_train, y_val, freeze=True, epochs=1000, lr=0.0001, augment_data=True)
