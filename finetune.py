import tensorflow as tf
from dataset import Dataset
from alexnet import AlexNet
import wget
import os.path

# load dataset from res/data, each subdirectory is a class
dataset = Dataset(os.path.join('res', 'data'), AlexNet.scale_size, mean_image=AlexNet.mean_image)

# input placeholders
in_images = tf.placeholder(tf.float32, [None, AlexNet.crop_size, AlexNet.crop_size, AlexNet.channels])
in_labels = tf.placeholder(tf.float32, [None, dataset.num_classes])

# instantiate network
net = AlexNet({'data': in_images})

# we are fine-tuning from ILSVRC, last IP layer is replaced with a fresh one
last_layer = net.layers['fc8new']

# define cost, accuracy and train operations
cost_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=in_labels), 0)

net_out = tf.argmax(tf.nn.softmax(last_layer), 1)
acc_op = tf.reduce_sum(tf.cast(tf.equal(net_out, tf.argmax(in_labels, 1)), tf.float32))
batch_size = tf.placeholder(tf.float32)
acc_op = tf.divide(acc_op, batch_size)

optimizer = tf.train.RMSPropOptimizer(0.001)

# retrieve test data
test_images, test_labels = dataset.get_augmented_epoch(AlexNet.crop_size, phase='test')

trainable_layers = tf.trainable_variables()

# weights from [https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet]
# converted with [https://github.com/ethereon/caffe-tensorflow]
weights = os.path.join('res', 'alexnet_caffemodel.npy')
if not os.path.exists(weights):
    wget.download('https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet_caffemodel.npy?dl=1')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # load weights, ignore weights for new layer
    print('Loading weights from: {0}'.format(weights))
    net.load(weights, session, ignore_missing=True)

    trainable_count = 0
    for epoch in range(100):
        # shuffle and randomly crop train dataset
        epoch_images, epoch_labels = dataset.get_augmented_epoch(AlexNet.crop_size, phase='train')

        for chunk in range(0, len(epoch_images), AlexNet.batch_size):
            # unlock new layer weights and biases
            if epoch % 1 == 0 and trainable_count * 2 < len(trainable_layers):
                trainable_count += 1
                print('layer: {0} now trainable'.format(trainable_layers[-2 * trainable_count].name.split('/')[0]))
                train_op = optimizer.minimize(cost_op, var_list=trainable_layers[-2 * trainable_count:])
                session.run(tf.variables_initializer(optimizer.variables()))

            # fetch batch images and labels
            batch_images = epoch_images[chunk:chunk + AlexNet.batch_size]
            batch_labels = epoch_labels[chunk:chunk + AlexNet.batch_size]

            # train
            session.run(train_op, feed_dict={in_images: batch_images,
                                             in_labels: batch_labels})

        # evaluate train performance
        train_loss, train_oa = session.run([cost_op, acc_op], feed_dict={in_images: batch_images,
                                                                         in_labels: batch_labels,
                                                                         batch_size: len(batch_labels)})

        # evaluate test performance
        test_loss, test_oa = session.run([cost_op, acc_op], feed_dict={in_images: test_images,
                                                                       in_labels: test_labels,
                                                                       batch_size: len(test_labels)})

        print('Epoch: {0} '
              'Train_OA: {1:.2f} '
              'Test_OA: {2:.2f} '
              'TrainLoss: {3:.2f} '
              'TestLoss: {4:.2f}'.format(epoch, train_oa, test_oa, train_loss, test_loss))
