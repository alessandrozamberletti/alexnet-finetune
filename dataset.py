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


class Dataset:
    def __init__(self, path, size, mean_image=None):
        self.path = path
        self.images = []
        self.labels = []
        self.size = size
        self.mean_image = mean_image
        self.num_classes = None

    def load_sample(self, whole_file_reader, files_queue):
        fn, content = whole_file_reader.read(files_queue)
        # load
        im = tf.image.decode_jpeg(content)
        # resize
        im = tf.image.resize_images(im,
                                    (self.size, self.size),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # rgb2bgr
        im = im[..., ::-1]
        # sub mean
        if self.mean_image is not None:
            im = im - self.mean_image

        return im, path2label(fn)

    def load(self):
        file_names = tf.train.match_filenames_once(self.path + '/*/*.jpg')
        files_num = tf.size(file_names)

        queue = tf.train.string_input_producer(file_names)
        reader = tf.WholeFileReader()

        loading_queue = tf.FIFOQueue(128, [tf.uint8, tf.string], None)
        load_sample_op = loading_queue.enqueue(self.load_sample(reader, queue))

        queue_runner = tf.train.QueueRunner(loading_queue,
                                            [load_sample_op]*24,
                                            loading_queue.close(),
                                            loading_queue.close(cancel_pending_enqueues=True))

        tf.train.add_queue_runner(queue_runner)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            num_samples = sess.run(files_num)
            for _ in range(num_samples):
                image, label = sess.run(loading_queue.dequeue())
                self.images.append(image)
                self.labels.append(label)

            coord.request_stop()
            coord.join(threads)

        self.labels = one_hot_encode(self.labels)
        self.num_classes = self.labels.shape[1]

        return np.array(self.images), np.array(self.labels)

