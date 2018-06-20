import tensorflow as tf
import random
from dataset import Dataset
from alexnet import AlexNet

tf.set_random_seed(42)
random.seed(42)

dataset = Dataset('data', AlexNet.scale_size, mean_image=AlexNet.mean_image)

in_images = tf.placeholder(tf.float32, [None, AlexNet.crop_size, AlexNet.crop_size, AlexNet.channels])
in_labels = tf.placeholder(tf.float32, [None, dataset.num_classes])
num_samples = tf.placeholder(tf.float32)

net = AlexNet({'data': in_images})

last_layer = net.layers['fc8new']
probabilities = tf.nn.softmax(last_layer)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=last_layer, labels=in_labels), 0)
accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(probabilities, 1), tf.argmax(in_labels, 1)), tf.float32)), num_samples)

trainable_layers = tf.trainable_variables()

test_im, test_lbl = dataset.get_augmented_epoch(AlexNet.crop_size, phase='test')
optimizer = tf.train.RMSPropOptimizer(0.001)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    net.load('weights.npy', session, ignore_missing=True)

    trainable_count = 0
    for i in range(1000):
        if i % 50 == 0:
            trainable_count += 1
            print('layer: {0} is now trainable'.format(trainable_layers[-2 * trainable_count].name.split('/')[0]))
            train_operation = optimizer.minimize(loss, var_list=trainable_layers[-2 * trainable_count:])
            session.run(tf.variables_initializer(optimizer.variables()))

        np_images, np_labels = dataset.get_augmented_epoch(AlexNet.crop_size, phase='train')
        batch_loss, train_oa, _ = session.run([loss, accuracy, train_operation], feed_dict={in_images: np_images,
                                                                                            in_labels: np_labels,
                                                                                            num_samples: len(np_labels)})

        test_oa = session.run(accuracy, feed_dict={in_images: test_im,
                                                   in_labels: test_lbl,
                                                   num_samples: len(test_lbl)})

        print('Iteration: {0} '
              'Train_OA: {1:.2f} '
              'Test_OA: {2:.2f} '
              'Loss: {3:.2f}'.format(i, train_oa, test_oa, batch_loss))
