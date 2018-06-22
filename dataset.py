import sklearn.preprocessing
import tensorflow as tf
import numpy as np


def one_hot_encode(labels):
    int_labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels)
    int_labels = int_labels.reshape(len(int_labels), 1)
    return sklearn.preprocessing.OneHotEncoder(sparse=False).fit_transform(int_labels)


def path2label(file_path):
    fn = tf.string_split([file_path], '/')
    fn_len = tf.cast(fn.dense_shape[1], tf.int32)
    return fn.values[fn_len - tf.constant(2, dtype=tf.int32)]


def load_sample(whole_file_reader, files_queue):
    fn, content = whole_file_reader.read(files_queue)
    return tf.image.decode_jpeg(content), path2label(fn)


class Dataset:
    def __init__(self, path, size, mean_image=None):
        self.path = path
        self.images = []
        self.labels = []
        self.size = size
        self.mean_image = mean_image
        self.num_classes = None

    def load(self):
        file_names = tf.train.match_filenames_once(self.path + '/*/*.jpg')
        files_num = tf.size(file_names)

        queue = tf.train.string_input_producer(file_names)
        reader = tf.WholeFileReader()
        coord = tf.train.Coordinator()

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            num_samples = sess.run(files_num)
            for _ in range(num_samples):
                image, label = sess.run(load_sample(reader, queue))
                image = sess.run(tf.image.resize_images(image, (self.size, self.size),
                                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
                # RGB to BGR
                image = image[..., ::-1]
                if self.mean_image is not None:
                    image = image - self.mean_image

                self.images.append(image)
                self.labels.append(label)

            coord.request_stop()
            coord.join(threads)

            self.labels = one_hot_encode(self.labels)
            self.num_classes = self.labels.shape[1]

        return np.array(self.images), np.array(self.labels)

