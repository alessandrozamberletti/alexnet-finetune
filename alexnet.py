import tensorflow as tf
import numpy as np

from network import Network


def augment(images, labels, crop_size, channels, random_crop=True):
    assert len(images) == len(labels)
    shuffle = np.random.permutation(len(images))
    images_placeholder = tf.placeholder(tf.float32, shape=images.shape)
    if random_crop:
        fn = tf.map_fn(lambda image: tf.random_crop(image, [crop_size, crop_size, channels]), images_placeholder)
    else:
        fn = tf.map_fn(lambda image: tf.image.resize_images(image, [crop_size, crop_size]), images_placeholder)
    feed = {images_placeholder: images[shuffle]}
    images = tf.Session().run(fn, feed_dict=feed)

    return np.array(images), np.array(labels[shuffle])


class AlexNet(Network):
    batch_size = 500
    scale_size = 256
    crop_size = 227
    channels = 3
    mean_image = [104., 117., 124.]

    def __init__(self, num_classes, weights):
        self.in_images = tf.placeholder(tf.float32, [None, AlexNet.crop_size, AlexNet.crop_size, AlexNet.channels])
        self.in_labels = tf.placeholder(tf.float32, [None, num_classes])
        self.weights = weights

        super(AlexNet, self).__init__({'data': self.in_images}, num_classes)

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

    def fit(self, x_train, x_test, y_train, y_test, epochs=100, augment_data=True):
        # to fine-tune we replace IP layer with a new one
        last_layer = self.layers['new']

        # define cost, accuracy and train operations
        cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=self.in_labels), 0)

        net_out = tf.argmax(tf.nn.softmax(last_layer), 1)
        acc_op = tf.reduce_sum(tf.cast(tf.equal(net_out, tf.argmax(self.in_labels, 1)), tf.float32))
        batch_size = tf.placeholder(tf.float32)
        acc_op = tf.divide(acc_op, batch_size)

        optimizer = tf.train.RMSPropOptimizer(0.001)

        # retrieve test data
        test_images, test_labels = augment(x_test, y_test, self.crop_size, self.channels, augment_data)

        trainable_layers = tf.trainable_variables()

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            # load weights, ignore weights for new layer
            self.load(self.weights, session, ignore_missing=True)

            trainable_count = 0
            for epoch in range(epochs):
                # unlock new layer weights and biases
                if epoch % 1 == 0 and trainable_count * 2 < len(trainable_layers):
                    trainable_count += 1
                    print('layer: "{0}" now trainable'.format(trainable_layers[-2 * trainable_count].name.split('/')[0]))
                    train_op = optimizer.minimize(cost_op, var_list=trainable_layers[-2 * trainable_count:])
                    session.run(tf.variables_initializer(optimizer.variables()))

                # augment train data
                epoch_images, epoch_labels = augment(x_train, y_train, self.crop_size, self.channels, augment_data)

                iteration = 0
                for chunk in range(0, len(epoch_images), AlexNet.batch_size):
                    # fetch batch images and labels
                    batch_images = epoch_images[chunk:chunk + AlexNet.batch_size]
                    batch_labels = epoch_labels[chunk:chunk + AlexNet.batch_size]

                    # evaluate train performance
                    feed = {self.in_images: batch_images, self.in_labels: batch_labels, batch_size: len(batch_labels)}
                    train_loss, train_oa, _ = session.run([cost_op, acc_op, train_op], feed_dict=feed)

                    # evaluate test performance
                    feed = {self.in_images: test_images, self.in_labels: test_labels, batch_size: len(test_labels)}
                    test_loss, test_oa = session.run([cost_op, acc_op], feed_dict=feed)

                    print('Epoch: {0} '
                          'Iteration: {1} '
                          'Train_OA: {2:.2f} '
                          'Test_OA: {3:.2f} '
                          'TrainLoss: {4:.2f} '
                          'TestLoss: {5:.2f}'.format(epoch, iteration, train_oa, test_oa, train_loss, test_loss))

                    iteration += 1
