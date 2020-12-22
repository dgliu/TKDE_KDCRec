import numpy as np
from tqdm import tqdm
import tensorflow as tf
import scipy.sparse as sparse
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix


class CausEMF(object):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 lamb=0.01,
                 lamb2=0.01,
                 learning_rate=1e-3,
                 optimizer=tf.train.AdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._num_users = num_users
        self._num_items = num_items
        self._double_num_items = num_items * 2
        self._embed_dim = embed_dim
        self._lamb = lamb
        self._lamb2 = lamb2
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('cause-mf'):
            # Placehoder
            self.user_idx = tf.placeholder(tf.int32, [None])
            self.item_idx = tf.placeholder(tf.int32, [None])
            self.label = tf.placeholder(tf.float32, [None])
            self.reg_idx = tf.placeholder(tf.int32, [None])

            # Variable to learn
            self.user_embeddings = tf.get_variable(name='users', shape=[self._num_users, self._embed_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.item_embeddings = tf.get_variable(name='items', shape=[self._double_num_items, self._embed_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.user_bias_embeddings = tf.get_variable(name='users_bias', shape=[self._num_users, ],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.item_bias_embeddings = tf.get_variable(name='items_bias', shape=[self._double_num_items, ],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

            with tf.variable_scope("factual_loss"):
                users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx)
                items = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx)
                user_bias = tf.nn.embedding_lookup(self.user_bias_embeddings, self.user_idx)
                item_bias = tf.nn.embedding_lookup(self.item_bias_embeddings, self.item_idx)

                x_ij = tf.reduce_sum(tf.multiply(users, items), axis=1) + user_bias + item_bias
                mf_loss = tf.reduce_mean(tf.square(self.label - x_ij))

                unique_user_idx, _ = tf.unique(self.user_idx)
                unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)

                unique_item_idx, _ = tf.unique(self.item_idx)
                unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)

                l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))

                self.factual_loss = mf_loss + self._lamb * l2_loss

            with tf.variable_scope("counter_factual_loss"):
                control_embed = tf.stop_gradient(
                    tf.nn.embedding_lookup(self.item_embeddings, self.reg_idx))
                self.cf_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.abs(tf.subtract(items, control_embed)), axis=1))

            with tf.variable_scope('loss'):
                self._loss = self.factual_loss + (self._lamb2 * self.cf_loss)

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
    def get_batches(user_item_pairs, matrix_train, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(user_item_pairs) / batch_size)):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(matrix_train[ui_pairs[:, 0], ui_pairs[:, 1]])[0]

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], label])

        return batches

    @staticmethod
    def compute_2i_regularization_id(prods, num_products):
        """Compute the ID for the regularization for the 2i approach"""

        reg_ids = []
        # Loop through batch and compute if the product ID is greater than the number of products
        for x in np.nditer(prods):
            if x >= num_products:
                reg_ids.append(x)
            elif x < num_products:
                reg_ids.append(x + num_products)  # Add number of products to create the 2i representation

        return np.asarray(reg_ids)

    def train_model(self, matrix_train, matrix_valid, epoch=100, metric='AUC', topK=50, is_topK=False):
        user_item_matrix = lil_matrix(matrix_train)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            batches = self.get_batches(user_item_pairs, matrix_train, self._batch_size)
            for step in range(len(batches)):
                reg_idx = self.compute_2i_regularization_id(batches[step][1], self._num_items)
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2],
                             self.reg_idx: reg_idx
                             }
                training = self.sess.run([self._train], feed_dict=feed_dict)

            RQ, Y, xBias, yBias = self.sess.run([self.user_embeddings,
                                                 self.item_embeddings[0:self._num_items, :],
                                                 self.user_bias_embeddings,
                                                 self.item_bias_embeddings[0:self._num_items]])
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


def causemf(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
            iteration=100, lam=0.01, lam2=0.01, rank=200, batch_size=500, learning_rate=1e-3, optimizer="LAdam", seed=0,
            gpu_on=False, metric='AUC', topK=50, is_topK=False, **unused):
    progress = WorkSplitter()

    progress.section("CausE-MF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("CausE-MF: Training")
    m, n = matrix_train.shape

    # Create new item IDs for S_t (i.e., [n, n*2)
    unif_user_item_matrix = lil_matrix(matrix_utrain)
    unif_user_item_pairs = np.asarray(unif_user_item_matrix.nonzero()).T
    unif_label = np.asarray(matrix_utrain[unif_user_item_pairs[:, 0], unif_user_item_pairs[:, 1]]).T
    unif_user_item_pairs[:, 1] += n

    # Create new csr matrix including union of S_c and S_t
    norm_user_item_matrix = lil_matrix(matrix_train)
    norm_user_item_pairs = np.asarray(norm_user_item_matrix.nonzero()).T
    norm_label = np.asarray(matrix_train[norm_user_item_pairs[:, 0], norm_user_item_pairs[:, 1]]).T

    user_item_pairs = np.vstack((unif_user_item_pairs, norm_user_item_pairs))
    labels = np.vstack((unif_label, norm_label))
    matrix_train = sparse.csr_matrix(
        (labels[:, 0], (user_item_pairs[:, 0], user_item_pairs[:, 1])), shape=(m, n * 2))

    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape

    model = CausEMF(m, n//2, rank, batch_size, lamb=lam, lamb2=lam2, learning_rate=learning_rate,
                    optimizer=Regularizer[optimizer], gpu_on=gpu_on)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_valid, iteration, metric, topK, is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias