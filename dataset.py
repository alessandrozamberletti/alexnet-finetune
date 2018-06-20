from sklearn.model_selection import train_test_split
import utils


class Dataset:
    def __init__(self, path, scale, mean_image=None, test_size=0.33):
        images, labels = utils.load_images(path, scale, mean_image)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(images, labels, test_size=test_size)
        self.num_classes = self.y_train.shape[1]

    @property
    def train(self):
        return self.X_train, self.y_train

    @property
    def test(self):
        return self.X_test, self.y_test

    def get_augmented_epoch(self, crop_size, phase='train'):
        if phase == 'test':
            return utils.augment(crop_size, self.X_test.shape[3], self.X_test, self.y_test)
        return utils.augment(crop_size, self.X_train.shape[3], self.X_train, self.y_train)

