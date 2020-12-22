import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix


class DelayAE(object):
    def __init__(self, input_dim, embed_dim, batch_size,
                 lamb=0.01,
                 step=3,
                 learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._input_dim = self._output_dim = input_dim
        self._embed_dim = embed_dim
        self._lamb = lamb
        self._step = step
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('delay-ae'):
            # Placehoder
            self.inputs = tf.placeholder(tf.float32, (None, self._input_dim))

            with tf.variable_scope('encode'):
                self.encode_weights = tf.Variable(
                    tf.truncated_normal([self._input_dim, self._embed_dim], stddev=1 / 500.0),
                    name="Weights")
                self.encode_bias = tf.Variable(tf.constant(0., shape=[self._embed_dim]), name="Bias")

                self.encoded = tf.nn.relu(tf.matmul(self.inputs, self.encode_weights) + self.encode_bias)

            with tf.variable_scope('decode'):
                self.decode_weights = tf.Variable(
                    tf.truncated_normal([self._embed_dim, self._output_dim], stddev=1 / 500.0),
                    name="Weights")
                self.decode_bias = tf.Variable(tf.constant(0., shape=[self._output_dim]), name="Bias")
                self.prediction = tf.matmul(self.encoded, self.decode_weights) + self.decode_bias

            with tf.variable_scope('loss'):
                l2_loss = tf.nn.l2_loss(self.encode_weights) + tf.nn.l2_loss(self.decode_weights)

                mask = tf.where(tf.not_equal(self.inputs, 0), tf.ones(tf.shape(self.inputs)),
                                tf.zeros(tf.shape(self.inputs)))
                mf_loss = tf.square(self.inputs - self.prediction * mask)

                self._loss = tf.reduce_mean(mf_loss) + self._lamb * tf.reduce_mean(l2_loss)

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)

            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            if self._gpu_on:
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.ConfigProto(device_count={'GPU': 0})
            self.sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            self.sess.run(init)

    @staticmethod
    def get_batches(rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def train_model(self, matrix_train, matrix_unif_train,  matrix_valid, epoch=100, metric='AUC', topK=50,
                    is_topK=False):
        batches = self.get_batches(matrix_train, self._batch_size)
        unif_batches = self.get_batches(matrix_unif_train, self._batch_size)
        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            if (i != 0) and (i % self._step == 0):
                for step in range(len(unif_batches)):
                    feed_dict = {self.inputs: unif_batches[step].todense()}
                    training = self.sess.run([self._train], feed_dict=feed_dict)
            else:
                for step in range(len(batches)):
                    feed_dict = {self.inputs: batches[step].todense()}
                    training = self.sess.run([self._train], feed_dict=feed_dict)

            RQ, X, xBias = self.get_RQ(matrix_train)
            Y = self.get_Y()
            yBias = self.get_yBias()
            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y.T,
                                                         topK=topK,
                                                         bias_V=yBias,
                                                         matrix_Train=matrix_train,
                                                         matrix_Test=matrix_valid,
                                                         is_topK=is_topK)
            result = evaluate(rating_prediction, topk_prediction, matrix_valid, [metric], [topK], is_topK=is_topK)

            if result[metric][0] > best_result:
                best_result = result[metric][0]
                best_RQ, best_X, best_Y, best_xBias, best_yBias = RQ, X, Y, xBias, yBias

        return best_RQ, best_X, best_xBias, best_Y, best_yBias

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self._batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ), self.sess.run(self.encode_weights), self.sess.run(self.encode_bias)

    def get_Y(self):
        return self.sess.run(self.decode_weights)

    def get_yBias(self):
        return self.sess.run(self.decode_bias)


def delayae(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
            iteration=100, lam=0.01, step=3, rank=200, batch_size=256, learning_rate=1e-4, optimizer="Adam", seed=0,
            gpu_on=False, metric='AUC', topK=50, is_topK=False, **unused):
    progress = WorkSplitter()

    progress.section("Delay-AE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Delay-AE: Training")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = DelayAE(n, rank, batch_size, lamb=lam, step=step, learning_rate=learning_rate,
                    optimizer=Regularizer[optimizer], gpu_on=gpu_on)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_utrain, matrix_valid, iteration, metric, topK,
                                               is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias