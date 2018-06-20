import tensorflow as tf
from dataset import Dataset
from alexnet import AlexNet
import wget
import os.path

data_dir = 'res'
weights = os.path.join(data_dir, 'alexnet_caffemodel.npy')
dataset_dir = os.path.join(data_dir, 'data')
epochs = 100

dataset = Dataset(dataset_dir, AlexNet.scale_size, mean_image=AlexNet.mean_image)

in_images = tf.placeholder(tf.float32, [None, AlexNet.crop_size, AlexNet.crop_size, AlexNet.channels])
in_labels = tf.placeholder(tf.float32, [None, dataset.num_classes])
num_samples = tf.placeholder(tf.float32)

net = AlexNet({'data': in_images})

last_layer = net.layers['fc8new']
probabilities = tf.nn.softmax(last_layer)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=in_labels), 0)
accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probabilities, 1), tf.argmax(in_labels, 1)), tf.float32)), num_samples)

trainable_layers = tf.trainable_variables()

test_im, test_lbl = dataset.get_augmented_epoch(AlexNet.crop_size, phase='test')
optimizer = tf.train.RMSPropOptimizer(0.001)

if not os.path.exists(weights):
    wget.download('https://www.dropbox.com/s/ekgz9jtj1ybtxmj/alexnet_caffemodel.npy?dl=1')

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    print('Loading weights from: {0}'.format(weights))
    net.load(weights, session, ignore_missing=True)

    trainable_count = 0
    for epoch in range(epochs):
        if epoch % 50 == 0:
            trainable_count += 1
            print('layer: {0} is now trainable'.format(trainable_layers[-2 * trainable_count].name.split('/')[0]))
            train_operation = optimizer.minimize(cost, var_list=trainable_layers[-2 * trainable_count:])
            session.run(tf.variables_initializer(optimizer.variables()))

        np_images, np_labels = dataset.get_augmented_epoch(AlexNet.crop_size, phase='train')
        loss, train_oa, _ = session.run([cost, accuracy, train_operation], feed_dict={in_images: np_images,
                                                                                      in_labels: np_labels,
                                                                                      num_samples: len(np_labels)})

        test_oa = session.run(accuracy, feed_dict={in_images: test_im,
                                                   in_labels: test_lbl,
                                                   num_samples: len(test_lbl)})

        print('Epoch: {0} '
              'Train_OA: {1:.2f} '
              'Test_OA: {2:.2f} '
              'Loss: {3:.2f}'.format(epoch, train_oa, test_oa, loss))
