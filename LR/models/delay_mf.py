import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix


class DelayMF(object):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 lamb=0.01,
                 step=3,
                 learning_rate=1e-3,
                 optimizer=tf.contrib.opt.LazyAdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._num_users = num_users
        self._num_items = num_items
        self._embed_dim = embed_dim
        self._lamb = lamb
        self._step = step
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('delay-mf'):
            # Placehoder
            self.user_idx = tf.placeholder(tf.int32, [None])
            self.item_idx = tf.placeholder(tf.int32, [None])
            self.label = tf.placeholder(tf.float32, [None])

            # Variable to learn
            self.user_embeddings = tf.get_variable(name='users', shape=[self._num_users, self._embed_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.item_embeddings = tf.get_variable(name='items', shape=[self._num_items, self._embed_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.user_bias_embeddings = tf.get_variable(name='users_bias', shape=[self._num_users, ],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.item_bias_embeddings = tf.get_variable(name='items_bias', shape=[self._num_items, ],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

            with tf.variable_scope("mf_loss"):
                users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx)
                users_bias = tf.nn.embedding_lookup(self.user_bias_embeddings, self.user_idx)
                items = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx)
                items_bias = tf.nn.embedding_lookup(self.item_bias_embeddings, self.item_idx)

                x_ij = tf.reduce_sum(tf.multiply(users, items), axis=1) + users_bias + items_bias

                mf_loss = tf.reduce_mean(tf.square(self.label - x_ij))

            with tf.variable_scope('l2_loss'):
                unique_user_idx, _ = tf.unique(self.user_idx)
                unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)

                unique_item_idx, _ = tf.unique(self.item_idx)
                unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)

                l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))

            with tf.variable_scope('loss'):
                self._loss = mf_loss + self._lamb * l2_loss

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

    def get_batches(self, user_item_pairs, unif_user_item_pairs, rating_matrix, matrix_unif_train, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(user_item_pairs) / batch_size)):
            if (i != 0) and (i % self._step == 0):
                unif_idx = np.random.choice(len(unif_user_item_pairs), batch_size, replace=False)
                unif_ui_pairs = unif_user_item_pairs[unif_idx, :]
                unif_ui_pairs = unif_ui_pairs.astype('int32')

                unif_label = np.asarray(matrix_unif_train[unif_ui_pairs[:, 0], unif_ui_pairs[:, 1]])[0]

                batches.append([unif_ui_pairs[:, 0], unif_ui_pairs[:, 1], unif_label])

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]])[0]

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], label])

        return batches

    def train_model(self, matrix_train, matrix_unif_train, matrix_valid, epoch=100, metric='AUC', topK=50,
                    is_topK=False):
        user_item_matrix = lil_matrix(matrix_train)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T
        unif_user_item_matrix = lil_matrix(matrix_unif_train)
        unif_user_item_pairs = np.asarray(unif_user_item_matrix.nonzero()).T

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            batches = self.get_batches(user_item_pairs, unif_user_item_pairs,
                                       matrix_train, matrix_unif_train, self._batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2]
                             }
                training = self.sess.run([self._train], feed_dict=feed_dict)

            RQ, Y, xBias, yBias = self.sess.run([self.user_embeddings,
                                                 self.item_embeddings,
                                                 self.user_bias_embeddings,
                                                 self.item_bias_embeddings])
            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y,
                                                         topK=topK,
                                                         bias_U=xBias,
                                                         bias_V=yBias,
                                                         matrix_Train=matrix_train,
                                                         matrix_Test=matrix_valid,
                                                         is_topK=is_topK)
            result = evaluate(rating_prediction, topk_prediction, matrix_valid, [metric], [topK], is_topK=is_topK)

            if result[metric][0] > best_result:
                best_result = result[metric][0]
                best_RQ, best_Y, best_xBias, best_yBias = RQ, Y, xBias, yBias

        return best_RQ, best_X, best_xBias, best_Y.T, best_yBias


def delaymf(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
            iteration=100, lam=0.01, step=3, rank=200, batch_size=500, learning_rate=1e-3, optimizer="LAdam", seed=0,
            gpu_on=False, metric='AUC', topK=50, is_topK=False, **unused):
    progress = WorkSplitter()

    progress.section("Delay-MF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Delay-MF: Training")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = DelayMF(m, n, rank, batch_size, lamb=lam, step=step, learning_rate=learning_rate,
                    optimizer=Regularizer[optimizer], gpu_on=gpu_on)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_utrain, matrix_valid, iteration, metric, topK,
                                               is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias

