import tensorflow as tf
import numpy as np

from network import Network


def process(images, labels, crop_size, channels, random_crop=True):
    assert len(images) == len(labels)
    shuffle = np.random.permutation(len(images))

    images_ph = tf.placeholder(tf.float32, shape=images.shape)
    if random_crop:
        fn = tf.map_fn(lambda image: tf.random_crop(image, [crop_size, crop_size, channels]), images_ph)
    else:
        fn = tf.map_fn(lambda image: tf.image.resize_images(image, [crop_size, crop_size]), images_ph)
    feed = {images_ph: images[shuffle]}
    images = tf.Session().run(fn, feed_dict=feed)

    return np.array(images), np.array(labels[shuffle])


class AlexNet(Network):
    BATCH_SIZE = 500
    SCALE_SIZE = 256
    CROP_SIZE = 227
    CHANNELS = 3
    MEAN_IMAGE = [104., 117., 124.]

    def __init__(self, num_classes, weights):
        self.num_classes = num_classes
        self.in_images_ph = tf.placeholder(tf.float32, [None, AlexNet.CROP_SIZE, AlexNet.CROP_SIZE, AlexNet.CHANNELS])
        self.in_labels_ph = tf.placeholder(tf.float32, [None, self.num_classes])
        self.weights = weights

        super(AlexNet, self).__init__({'data': self.in_images_ph}, self.num_classes)

    def setup(self):
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
         .lrn(2, 2e-05, 0.75, name='norm1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
         .lrn(2, 2e-05, 0.75, name='norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 384, 1, 1, name='conv3')
         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7')
         .fc(self.num_classes, relu=False, name='new')
         .softmax(name='prob'))

    def __define_ops(self, lr):
        # weights for last fc layer are ignored
        last_layer = self.layers['new']

        # define cost, accuracy and train operations
        cost_op = tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=self.in_labels_ph)
        self.cost_op = tf.reduce_mean(cost_op, 0)

        net_out = tf.argmax(tf.nn.softmax(last_layer), 1)
        acc_op = tf.reduce_sum(tf.cast(tf.equal(net_out, tf.argmax(self.in_labels_ph, 1)), tf.float32))
        self.batch_size_ph = tf.placeholder(tf.float32)
        self.acc_op = tf.divide(acc_op, self.batch_size_ph)

        self.optimizer = tf.train.RMSPropOptimizer(lr)

    def fit(self, x_train, x_val, y_train, y_val, freeze=True, epochs=100, augment=True, lr=0.001):
        self.__define_ops(lr)

        # validation data
        val_images, val_labels = process(x_val, y_val, self.CROP_SIZE, self.CHANNELS, random_crop=False)

        trainable_layers = tf.trainable_variables()
        if not freeze:
            print('*** all layers will be trained ***')
            train_op = self.optimizer.minimize(self.cost_op, var_list=trainable_layers)
        else:
            print('*** decremental fine-tuning will be performed ***')

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # load weights, ignore weights for new layer
            self.load(self.weights, session, ignore_missing=True)

            trainable_count = 0
            for epoch in range(epochs):
                # unlock new layers
                if freeze and epoch % 10 == 0 and trainable_count * 2 < len(trainable_layers):
                    trainable_count += 1
                    layer_name = trainable_layers[-2 * trainable_count].name.split('/')[0]
                    print('*** layer ({0}) is now trainable ***'.format(layer_name))
                    train_op = self.optimizer.minimize(self.cost_op, var_list=trainable_layers[-2*trainable_count:])
                    session.run(tf.variables_initializer(self.optimizer.variables()))

                # augment
                epoch_images, epoch_labels = process(x_train, y_train, self.CROP_SIZE, self.CHANNELS, random_crop=augment)

                iteration = 0
                for batch_start in range(0, len(epoch_images), AlexNet.BATCH_SIZE):
                    # fetch batch images and labels
                    batch_images = epoch_images[batch_start:batch_start+AlexNet.BATCH_SIZE]
                    batch_labels = epoch_labels[batch_start:batch_start+AlexNet.BATCH_SIZE]

                    # train performance
                    feed = {self.in_images_ph: batch_images,
                            self.in_labels_ph: batch_labels,
                            self.batch_size_ph: len(batch_labels)}
                    train_loss, train_oa, _ = session.run([self.cost_op, self.acc_op, train_op], feed_dict=feed)

                    # validation performance
                    feed = {self.in_images_ph: val_images,
                            self.in_labels_ph: val_labels,
                            self.batch_size_ph: len(val_labels)}
                    val_loss, val_oa = session.run([self.cost_op, self.acc_op], feed_dict=feed)

                    print('Epoch: {0} '
                          'Iteration: {1} '
                          'Train_OA: {2:.2f} '
                          'Val_OA: {3:.2f} '
                          'TrainLoss: {4:.2f} '
                          'ValLoss: {5:.2f}'.format(epoch, iteration, train_oa, val_oa, train_loss, val_loss))

                    iteration += 1
