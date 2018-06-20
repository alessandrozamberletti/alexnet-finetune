import os.path
import wget

from sklearn.model_selection import train_test_split
from dataset import Dataset
from alexnet import AlexNet


# load dataset
data = Dataset('res/data', AlexNet.scale_size, mean_image=AlexNet.mean_image)
images, labels = data.load()

# split train-test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=.33)

# download ILSVRC weights
weights_file = 'res/alexnet_caffemodel.npy'
if not os.path.exists(weights_file):
    wget.download('https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet_caffemodel.npy?dl=1', 'res')

# initialize network
alex_net = AlexNet(data.num_classes, weights_file)

# train
alex_net.fit(X_train, X_test, y_train, y_test, epochs=100, augment_data=False)
