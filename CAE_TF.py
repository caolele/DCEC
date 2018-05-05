import os, sys
import random
import tensorflow as tf
from model_wrapper import define_scope
from datasets import load_mnist
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants

tf.app.flags.DEFINE_integer('epoch', 2, 'number of epochs.')
tf.app.flags.DEFINE_integer('ver', 1, 'version number of the model.')
tf.app.flags.DEFINE_integer('bs', 256, 'batch size of training.')
tf.app.flags.DEFINE_float('lr', 0.001, 'initial learning rate.')
tf.app.flags.DEFINE_string('dir', './dump', 'Working directory.')
FLAGS = tf.app.flags.FLAGS


class Model:

    def __init__(self, filters=[32, 64, 128, 10], init_lr=0.001, input_shape=(None, 28, 28, 1)):
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
        self.loss
        self.optimize

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
    def loss(self):
        return tf.losses.mean_squared_error(labels=self.label, predictions=self.cae)

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.minimize(self.loss, global_step=self.step)


def main(_):
    if len(sys.argv) < 3 or not sys.argv[1].startswith('--epoch') or not sys.argv[2].startswith('--ver'):
        print('Usage: upf_saved_model.py --epoch=a --ver=b [--bs=c --lr=d --dir=e]')
        sys.exit(-1)
    if FLAGS.epoch <= 0 or FLAGS.ver <= 0 or FLAGS.bs <= 0 or FLAGS.lr <= 0:
        print('training epoch, model version, batch size, and learning rate should be positive!')
        sys.exit(-1)

    print('Epoch:%d | Version:%d | BatchSize:%d | LeaningRate:%f | Directory:%s' %
          (FLAGS.epoch, FLAGS.ver, FLAGS.bs, FLAGS.lr, FLAGS.dir))

    export_path = os.path.join(FLAGS.dir, str(FLAGS.ver))

    # load dataset
    x, y = load_mnist()
    n_train = x.shape[0]
    print("Shape of x:", x.shape)
    print("Shape of y:", y.shape)

    # initialize model
    input_shape = (None, x.shape[1], x.shape[2], x.shape[3])
    model = Model(filters=[32, 64, 128, 10], init_lr=FLAGS.lr, input_shape=input_shape)
    init_op = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init_op)

    # training
    n_batches = int(n_train / FLAGS.bs)
    disp_freq = 20
    for i in range(FLAGS.epoch):
        aidx = list(range(n_train))
        random.shuffle(aidx)

        ptr, loss = 0, 0
        print('Training: initLr = %f, batchSize = %d' % (FLAGS.lr, FLAGS.bs))
        sys.stdout.flush()
        for j in range(n_batches):
            inp = x[aidx[ptr:ptr + FLAGS.bs], :, :, :]
            ptr += FLAGS.bs

            _, _ce, _lr = sess.run([model.optimize, model.loss, model.learning_rate],
                                   feed_dict={model.data: inp, model.label: inp})

            if (j + 1) % disp_freq == 0:
                print('Epoch %d/%d Batch %d/%d: loss = %f , lr = %f'
                      % (i + 1, FLAGS.epoch, j + 1, n_batches, loss / disp_freq, _lr))
                sys.stdout.flush()
                loss = 0
            else:
                loss += _ce

    print('Exporting trained model to', export_path)
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING])
    builder.save(True)

    sess.close()


if __name__ == '__main__':
    tf.app.run()
