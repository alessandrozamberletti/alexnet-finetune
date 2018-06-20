import os
import cv2
import numpy as np
import tensorflow as tf
import sklearn.preprocessing


def load_images(path, scale, mean):
    images = []
    labels = []
    for class_name in os.listdir(path):
        if os.path.isdir(os.path.join(path, class_name)):
            class_dir = os.path.join(path, class_name)
            class_images, class_labels = load_class(class_dir, class_name, scale, mean)
            images += class_images
            labels += class_labels
    return np.array(images), np.array(one_hot_encode(labels))


def augment(crop_size, channels, images, labels):
    assert len(images) == len(labels)
    shuffle = np.random.permutation(len(images))
    placeholder = tf.placeholder(tf.float32, shape=images.shape)
    fn = tf.map_fn(lambda image: tf.random_crop(image, [crop_size, crop_size, channels]), placeholder)
    return np.array(tf.Session().run(fn, feed_dict={placeholder: images[shuffle]})), np.array(labels[shuffle])


def load_class(path, class_name, scale, mean):
    images = []
    labels = []
    image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith(".jpg")]
    for filename in image_files:
        im_path = os.path.join(path, filename)
        image = cv2.imread(im_path)
        image = cv2.resize(image, dsize=(scale, scale))

        images.append(image - mean)
        labels.append(class_name)
    return images, labels


def one_hot_encode(labels):
    int_labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels)
    int_labels = int_labels.reshape(len(int_labels), 1)
    return sklearn.preprocessing.OneHotEncoder(sparse=False).fit_transform(int_labels)
