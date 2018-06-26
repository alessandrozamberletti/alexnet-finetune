import os.path
import wget

from sklearn.model_selection import train_test_split
from dataset import Dataset
from alexnet import AlexNet


print('Loading data..')
images, labels = Dataset('res/data1', AlexNet.SCALE_SIZE, mean_image=AlexNet.MEAN_IMAGE).load()

print('Splitting into train&validation..')
X_train, X_val, y_train, y_val = train_test_split(images, labels)

print('Looking for pre-trained weights..')
weights_file = 'res/alexnet-caffemodel.npy'
if not os.path.exists(weights_file):
    wget.download('https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet-caffemodel.npy?dl=1', 'res')

print('Fine-tuning AlexNet')
alex_net = AlexNet(labels.shape[1], weights_file)
alex_net.fit(X_train, X_val, y_train, y_val, freeze=True, epochs=1000, lr=0.001)
