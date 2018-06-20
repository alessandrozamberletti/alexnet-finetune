import os
import numpy as np
import cv2
import sklearn.preprocessing


class Dataset:
    def __init__(self, path, scale, mean_image=None):
        self.path = path
        self.scale = scale
        self.mean_image = mean_image
        self.images = []
        self.labels = []
        self.num_classes = None

    def load(self):
        for class_name in os.listdir(self.path):
            if os.path.isdir(os.path.join(self.path, class_name)):
                class_dir = os.path.join(self.path, class_name)
                self.__load_class(class_dir, class_name)

        one_hot_labels = self.__one_hot_encode()
        self.num_classes = one_hot_labels.shape[1]
        return np.array(self.images), np.array(one_hot_labels)

    def __load_class(self, class_dir, class_name):
        image_files = [f for f in os.listdir(class_dir)
                       if os.path.isfile(os.path.join(class_dir, f)) and f.endswith(".jpg")]
        for image_file in image_files:
            im_path = os.path.join(class_dir, image_file)
            image = cv2.imread(im_path)

            image = cv2.resize(image, dsize=(self.scale, self.scale))
            if self.mean_image is not None:
                image = image - self.mean_image

            self.images.append(image)
            self.labels.append(class_name)

    def __one_hot_encode(self):
        int_labels = sklearn.preprocessing.LabelEncoder().fit_transform(self.labels)
        int_labels = int_labels.reshape(len(int_labels), 1)
        return sklearn.preprocessing.OneHotEncoder(sparse=False).fit_transform(int_labels)
