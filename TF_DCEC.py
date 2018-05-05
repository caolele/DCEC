import os
import math
import random
import metrics
import numpy as np
import tensorflow as tf
from model_wrapper import define_scope
from datasets import load_mnist
from progress_bar import progress
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants

tf.app.flags.DEFINE_integer('iterCAE', 210000, 'number of iterations to pretrain CAE mse loss')
tf.app.flags.DEFINE_integer('iterDCEC', 700000, 'number of iterations to jointly fine-tune DCEC loss')
tf.app.flags.DEFINE_integer('updateIntervalDCEC', 140, 'the update interval of training DCEC')
tf.app.flags.DEFINE_integer('kmeansTrainSteps', 1000, 'the number of steps to train Kmeans')
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
        self.t_dist = tf.placeholder(tf.float32, [None, cluster], name='kld_label')
        self.bs = tf.shape(self.data)
        self.step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(init_lr, self.step, 500, 0.8)
        self.code = None
        self.cae

        self.cur_iter = 0
        self.pretrain = pretrain
        self.cluster = cluster
        self.kmeans_input = tf.placeholder(tf.float32, (None, self.filters[3]), name='kmeans_input')
        self.all_scores, self.cluster_idx, self.scores, self.cluster_centers_initialized, \
            self.cluster_centers_var, self.init_op, self.training_op = self.kmeans
        self.avg_distance = tf.reduce_mean(self.all_scores)
        self.cluster_weights = tf.Variable(tf.zeros([cluster, filters[-1]]), dtype=tf.float32)
        # self.cluster_weights = tf.identity(self.cluster_centers_var)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.q_dist
        self.p_dist
        self.loss_cae
        self.loss_kld
        self.optimize_cae
        self.optimize

    def increment_cur_iter(self):
        self.cur_iter += 1

    @define_scope
    def assign_cluster_weights(self):
        return tf.assign(self.cluster_weights, self.cluster_centers_var)

    @define_scope
    def cae(self):
        cae_encoder = tf.layers.conv2d(self.data, self.filters[0], 5, strides=(2, 2),
                                       padding='same', activation=tf.nn.relu)
        cae_encoder = tf.layers.conv2d(cae_encoder, self.filters[1], 5, strides=(2, 2),
                                       padding='same', activation=tf.nn.relu)
        cae_encoder = tf.layers.conv2d(cae_encoder, self.filters[2], 3, strides=(2, 2),
                                       padding=self.pad3, activation=tf.nn.relu)
        dim_before_flatten = tf.shape(cae_encoder)
        cae_encoder = tf.layers.Flatten()(cae_encoder)
        flatten_dim = self.filters[2] * int(self.input_shape[1]/8) * int(self.input_shape[1]/8)
        self.code = tf.layers.dense(cae_encoder, self.filters[3])
        cae_decoder = tf.layers.dense(self.code, flatten_dim, activation=tf.nn.relu)
        cae_decoder = tf.reshape(cae_decoder, shape=dim_before_flatten)
        cae_decoder = tf.layers.Conv2DTranspose(self.filters[1], 3, strides=(2, 2),
                                                padding=self.pad3, activation=tf.nn.relu)(cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(self.filters[0], 5, strides=(2, 2),
                                                padding='same', activation=tf.nn.relu)(cae_decoder)
        cae_decoder = tf.layers.Conv2DTranspose(self.input_shape[-1], 5, strides=(2, 2),
                                                padding='same', activation=tf.nn.relu)(cae_decoder)
        return cae_decoder

    @define_scope
    def kmeans(self):
        return tf.contrib.factorization.KMeans(self.kmeans_input, num_clusters=self.cluster,
                                               use_mini_batch=True, initial_clusters='kmeans_plus_plus',
                                               ).training_graph()

    @define_scope
    def loss_cae(self):
        return tf.losses.mean_squared_error(labels=self.label, predictions=self.cae)

    @define_scope
    def loss_kld(self):
        # t_dist should NOT be re-calculated frequently to avoid instability
        return tf.reduce_sum(self.t_dist * tf.log(self.t_dist/self.q_dist))

    @define_scope
    def q_dist(self):
        # here we assume alpha of student t-distribution is 1.0
        q = 1.0 / (1.0 + tf.reduce_sum(
            tf.square(tf.expand_dims(self.code, 1) - self.cluster_weights), axis=-1))
        return tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=-1))

    @define_scope
    def p_dist(self):
        w = tf.square(self.q_dist) / tf.reduce_sum(self.q_dist, axis=0)
        return tf.transpose(tf.transpose(w) / tf.reduce_sum(w, axis=-1))

    @define_scope
    def optimize_cae(self):
        return self.optimizer.minimize(self.loss_cae, global_step=self.step)

    @define_scope
    def optimize(self):
        return self.optimizer.minimize(self.loss_cae + 0.1 * self.loss_kld, global_step=self.step)


def main(_):
    print("Using params:", FLAGS.__flags)

    # load dataset
    x, y = load_mnist() # You have to implement your own data feed function
    n_train = x.shape[0]

    # initialize model
    input_shape = (None, x.shape[1], x.shape[2], x.shape[3])
    model = Model(filters=[32, 64, 128, 10], init_lr=FLAGS.lr, input_shape=input_shape, cluster=FLAGS.cluster)
    init_param = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_param)

    print('CAE Pre-training ...')
    n_batches = int(n_train / FLAGS.bs)
    epoch = math.ceil(FLAGS.iterCAE / n_train)
    aidx = list(range(n_train))
    for i in range(epoch):
        random.shuffle(aidx)
        ptr, loss = 0, 0
        for j in range(n_batches):
            inp = x[aidx[ptr:ptr + FLAGS.bs], :, :, :]
            ptr += FLAGS.bs
            _, _ce, _lr = sess.run([model.optimize_cae, model.loss_cae, model.learning_rate],
                                   feed_dict={model.data: inp, model.label: inp})
            progress(j+1, n_batches, status=' CAE_Loss=%f, Lr=%f, Epoch=%d/%d' % (_ce, _lr, i+1, epoch))

    print('Initializing and Training K-means ...')
    init_codes = np.asarray(sess.run([model.code], feed_dict={model.data: x}))[0]
    while True:
        sess.run(model.init_op, feed_dict={model.kmeans_input: init_codes})
        isInit = sess.run(model.cluster_centers_initialized)
        if isInit:
            break
    for i in range(0, FLAGS.kmeansTrainSteps):
        _, d = sess.run([model.training_op, model.avg_distance], feed_dict={model.kmeans_input: init_codes})
        progress(i+1, FLAGS.kmeansTrainSteps, status=' Avg.Distance=%f, Step=%d' % (d, i+1))
    cidx, = sess.run([model.cluster_idx], feed_dict={model.kmeans_input: init_codes})[0]
    last_cidx = np.copy(cidx)
    sess.run([model.assign_cluster_weights])

    print('DCEC Fine-tuning ...')
    epoch = math.ceil(FLAGS.iterDCEC / n_train)
    for i in range(epoch):
        random.shuffle(aidx)
        ptr, loss = 0, 0

        # Check stop criterion
        delta_label = np.sum(cidx != last_cidx).astype(np.float32) / cidx.shape[0]
        last_cidx = np.copy(cidx)
        if i > 0 and delta_label < 0.001:
            print('Reached tolerance threshold. Stop training.')
            break

        # TODO: More evaluation metrics
        print('Epoch:%d, Accuracy:%f, Scale_of_label_change:%f' %
              (i, np.round(metrics.acc(y, cidx), 5), delta_label))

        for j in range(n_batches):
            if (i * n_batches + j) % FLAGS.updateIntervalDCEC == 0:
                # update the t-dist using all embeddings every updateIntervalDCEC iters
                p, q = sess.run([model.p_dist, model.q_dist], feed_dict={model.data: x})
                cidx = q.argmax(-1)

            # train on one batch
            inp = x[aidx[ptr:ptr + FLAGS.bs], :, :, :]
            pnp = p[aidx[ptr:ptr + FLAGS.bs], :]
            ptr += FLAGS.bs

            _, loss_cae, loss_kld, _lr = sess.run([model.optimize, model.loss_cae, model.loss_kld, model.learning_rate],
                                                  feed_dict={model.data: inp, model.t_dist: pnp, model.label: inp})
            progress(j + 1, n_batches, status=' CAE_Loss=%f, KL_Loss=%f, Loss=%f, Lr=%f, Epoch=%d/%d'
                                              % (loss_cae, loss_kld, loss_cae+0.1*loss_kld, _lr, i + 1, epoch))

    export_path = os.path.join(FLAGS.dir, str(FLAGS.ver))
    print('Exporting trained model to', export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING])
    builder.save(True)

    sess.close()


if __name__ == '__main__':
    tf.app.run()
