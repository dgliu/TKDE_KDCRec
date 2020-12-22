import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix, csr_matrix


class WeightCAE(object):
    def __init__(self, input_dim, embed_dim, batch_size, num_users,
                 lamb=0.01,
                 learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._input_dim = self._output_dim = input_dim
        self._embed_dim = embed_dim
        self._lamb = lamb
        self._batch_size = batch_size
        self.num_users = num_users
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('weightc-ae'):
            # Placehoder
            self.inputs = tf.placeholder(tf.float32, (None, self._input_dim))
            self.mask = tf.placeholder(tf.float32, (None, self._input_dim))
            self.sample_idx = tf.placeholder(tf.int32, [None])

            self.confidence = tf.get_variable(name='confidence', shape=[self.num_users, self._input_dim],
                                              initializer=tf.constant_initializer(0.3))

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

                confidence = tf.nn.embedding_lookup(self.confidence, self.sample_idx)
                clip_confidence = tf.clip_by_value(confidence, 0, 1)

                self._loss = tf.reduce_mean(
                    clip_confidence * self.mask * mf_loss + (1 - self.mask) * mf_loss) + self._lamb * tf.reduce_mean(
                    l2_loss)

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
    def get_batches(rating_matrix, marks, batch_size):
        remaining_size = rating_matrix.shape[0]
        index = np.arange(remaining_size)
        batch_index = 0
        batches = []
        marks_batches = []
        sample_idx = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
                marks_batches.append(marks[batch_index * batch_size:])
                sample_idx.append(index[batch_index * batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
                marks_batches.append(marks[batch_index * batch_size:(batch_index + 1) * batch_size])
                sample_idx.append(index[batch_index * batch_size:(batch_index + 1) * batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches, marks_batches, sample_idx

    def train_model(self, matrix_train, marks, matrix_valid, epoch=100, metric='AUC', topK=50, is_topK=False):
        batches, marks_batches, sample_idx = self.get_batches(matrix_train, marks, self._batch_size)

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense(),
                             self.mask: marks_batches[step].todense(),
                             self.sample_idx: sample_idx[step]}
                training = self.sess.run([self._train], feed_dict=feed_dict)

            RQ, X, xBias = self.get_RQ(matrix_train, marks)
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

    def get_RQ(self, rating_matrix, marks):
        batches, _, _ = self.get_batches(rating_matrix, marks, self._batch_size)
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


def oldweightcae(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
                 iteration=100, lam=0.01, rank=200, batch_size=256, learning_rate=1e-4, optimizer="Adam",
                 seed=0, gpu_on=False, metric='AUC', topK=50, is_topK=False, **unused):
    progress = WorkSplitter()

    progress.section("Old-WeightC-AE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Old-WeightC-AE: Training")
    marks = csr_matrix(matrix_train.shape)
    marks[(matrix_train != 0).nonzero()] = 1

    matrix_input = matrix_train + matrix_utrain
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = WeightCAE(n, rank, batch_size, m, lamb=lam, learning_rate=learning_rate, optimizer=Regularizer[optimizer],
                      gpu_on=gpu_on)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, marks, matrix_valid, iteration, metric, topK, is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias