import os
import sys
import math
import random
import numpy as np
import tensorflow as tf
from model_wrapper import define_scope
from datasets import load_mnist
from progress_bar import progress
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants

tf.app.flags.DEFINE_integer('iterCAE', 1000, 'number of iterations to pretrain CAE mse loss')
tf.app.flags.DEFINE_integer('iterDCEC', 500, 'number of iterations to jointly fine-tune DCEC loss')
tf.app.flags.DEFINE_integer('updateIntervalDCEC', 100, 'the update interval of training DCEC')
tf.app.flags.DEFINE_integer('cluster', 10, 'number of clusters')
tf.app.flags.DEFINE_integer('ver', 1, 'version number of the model.')
tf.app.flags.DEFINE_integer('bs', 256, 'batch size of training.')
tf.app.flags.DEFINE_float('lr', 0.001, 'initial learning rate.')
tf.app.flags.DEFINE_string('dir', './dump', 'Working directory.')

FLAGS = tf.app.flags.FLAGS


class Model:

    def __init__(self, filters=[32, 64, 128, 10], input_shape=(None, 28, 28, 1), init_lr=0.001,
                 pretrain=70000, cluster=10):
        self.filters = filters
        self.input_shape = input_shape
        if input_shape[1] % 8 == 0:
            self.pad3 = 'same'
        else:
            self.pad3 = 'valid'
        self.data = tf.placeholder(tf.float32, input_shape, name='input_data')
        self.label = tf.placeholder(tf.float32, input_shape, name='label')
        self.bs = tf.shape(self.data)
        self.step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(init_lr, self.step, 5000, 0.9)
        self.code = None
        self.cae
        self.loss_cae
        self.optimize

        self.cur_iter = 0
        self.pretrain = pretrain
        self.cluster = cluster
        self.kmeans

    def increment_cur_iter(self):
        self.cur_iter += 1

    @define_scope
    def cae(self):
        cae_encoder = tf.layers.conv2d(self.data, self.filters[0], 5, strides=(2, 2), padding='same', activation=tf.nn.relu)
        cae_encoder = tf.layers.conv2d(cae_encoder, self.filters[1], 5, strides=(2, 2), padding='same', activation=tf.nn.relu)
        cae_encoder = tf.layers.conv2d(cae_encoder, self.filters[2], 3, strides=(2, 2), padding=self.pad3, activation=tf.nn.relu)
        dim_before_flatten = tf.shape(cae_encoder)
        cae_encoder = tf.layers.Flatten()(cae_encoder)
        flatten_dim = self.filters[2] * int(self.input_shape[1]/8) * int(self.input_shape[1]/8)
        self.code = tf.layers.dense(cae_encoder, self.filters[3])
        cae_decoder = tf.layers.dense(self.code, flatten_dim, activation=tf.nn.relu)
        cae_decoder = tf.reshape(cae_decoder, shape=dim_before_flatten)
        cae_decoder = tf.layers.Conv2DTranspose(self.filters[1], 3, strides=(2, 2), padding=self.pad3, activation=tf.nn.relu)(cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(self.filters[0], 5, strides=(2, 2), padding='same', activation=tf.nn.relu)(cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(self.input_shape[-1], 5, strides=(2, 2), padding='same', activation=tf.nn.relu)(cae_decoder)
        return cae_decoder

    @define_scope
    def kmeans(self):
        return tf.contrib.factorization.KMeansClustering(num_clusters=self.cluster)

    @define_scope
    def loss_cae(self):
        return tf.losses.mean_squared_error(labels=self.label, predictions=self.cae)


    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss_cae, global_step=self.step)


def main(_):
    print("Using params:", FLAGS.__flags)

    # load dataset
    x, y = load_mnist()
    x = x[:2000,:,:,:]
    n_train = x.shape[0]

    # initialize model
    input_shape = (None, x.shape[1], x.shape[2], x.shape[3])
    model = Model(filters=[32, 64, 128, 10], init_lr=FLAGS.lr, input_shape=input_shape)
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)

    print('CAE Pre-training ...')
    n_batches = int(n_train / FLAGS.bs)
    epoch = math.ceil(FLAGS.iterCAE / n_train)
    for i in range(epoch):
        aidx = list(range(n_train))
        random.shuffle(aidx)
        ptr, loss = 0, 0
        sys.stdout.flush()
        for j in range(n_batches):
            inp = x[aidx[ptr:ptr + FLAGS.bs], :, :, :]
            ptr += FLAGS.bs
            _, _ce, _lr = sess.run([model.optimize, model.loss_cae, model.learning_rate],
                                   feed_dict={model.data: inp, model.label: inp})
            progress(j+1, n_batches, status=' Loss=%f, Epoch=%d' % (_ce, i+1))

    print('Initializing K-means ...')
    kmflow = sess.run([model.code], feed_dict={model.data: x})
    kmflow = np.asarray(kmflow)[0]
    model.kmeans.train(tf.estimator.inputs.numpy_input_fn({'x': kmflow}, shuffle=True, num_epochs=10))
    _kmpred = model.kmeans.predict_cluster_index(
        tf.estimator.inputs.numpy_input_fn({'x': kmflow}, shuffle=True))
    kmpred = []
    for p in _kmpred:
        kmpred.append(p)
    print(np.asarray(kmpred))
    exit(0)

    print('DCEC Fine-tuning ...')
    for i in range(FLAGS.iterDCEC):
        if i % FLAGS.updateIntervalDCEC == 0:
            pass #TODO: update step

        aidx = list(range(n_train))
        random.shuffle(aidx)



if __name__ == '__main__':
    tf.app.run()
