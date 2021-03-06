import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix, csr_matrix


class BridgeMF(object):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 init_unif_U, init_unif_V, init_unif_uBias, init_unif_vBias,
                 lamb=0.01,
                 lamb2=0.01,
                 learning_rate=1e-3,
                 measure='sigmoid',
                 optimizer=tf.contrib.opt.LazyAdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._num_users = num_users
        self._num_items = num_items
        self._embed_dim = embed_dim
        self.unif_U = init_unif_U
        self.unif_V = init_unif_V
        self.unif_uBias = init_unif_uBias
        self.unif_vBias = init_unif_vBias
        self._lamb = lamb
        self._lamb2 = lamb2
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._measure = measure
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('bridge-mf'):
            # Placehoder
            self.user_idx = tf.placeholder(tf.int32, [None])
            self.item_idx = tf.placeholder(tf.int32, [None])
            self.label = tf.placeholder(tf.float32, [None])

            self.mark = tf.placeholder(tf.float32, [None])

            self.sampled_user_idx = tf.placeholder(tf.int32, [None])
            self.sampled_item_idx = tf.placeholder(tf.int32, [None])

            # Variable to learn
            self.user_embeddings = tf.get_variable(name='users', shape=[self._num_users, self._embed_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.item_embeddings = tf.get_variable(name='items', shape=[self._num_items, self._embed_dim],
                                                   initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.user_bias_embeddings = tf.get_variable(name='users_bias', shape=[self._num_users, ],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.item_bias_embeddings = tf.get_variable(name='items_bias', shape=[self._num_items, ],
                                                        initializer=tf.truncated_normal_initializer(stddev=0.01))

            # Import Pre-Trained Variables
            self.unif_user_embeddings = tf.get_variable(name='unif_users', initializer=self.unif_U,
                                                        trainable=False)
            self.unif_item_embeddings = tf.get_variable(name='unif_items', initializer=self.unif_V,
                                                        trainable=False)
            self.unif_user_bias_embeddings = tf.get_variable(name='unif_users_bias', initializer=self.unif_uBias,
                                                             trainable=False)
            self.unif_item_bias_embeddings = tf.get_variable(name='unif_items_bias', initializer=self.unif_vBias,
                                                             trainable=False)

            with tf.variable_scope('factual_loss'):
                users = tf.nn.embedding_lookup(self.user_embeddings, self.user_idx)
                users_bias = tf.nn.embedding_lookup(self.user_bias_embeddings, self.user_idx)
                items = tf.nn.embedding_lookup(self.item_embeddings, self.item_idx)
                items_bias = tf.nn.embedding_lookup(self.item_bias_embeddings, self.item_idx)

                x_ij = tf.reduce_sum(tf.multiply(users, items), axis=1) + users_bias + items_bias

                unif_users = tf.nn.embedding_lookup(self.unif_user_embeddings, self.user_idx)
                unif_items = tf.nn.embedding_lookup(self.unif_item_embeddings, self.item_idx)

                user_cos = tf.reduce_sum(tf.multiply(users, unif_users), axis=1) / (
                            tf.sqrt(tf.reduce_sum(tf.square(users), axis=1)) *
                            tf.sqrt(tf.reduce_sum(tf.square(unif_users), axis=1)))
                item_cos = tf.reduce_sum(tf.multiply(items, unif_items), axis=1) / (
                            tf.sqrt(tf.reduce_sum(tf.square(items), axis=1)) *
                            tf.sqrt(tf.reduce_sum(tf.square(unif_items), axis=1)))

                if self._measure == 'exp':
                    confidence = tf.exp((user_cos + item_cos) / 2)
                    confidence = tf.clip_by_value(confidence, 0, 1)
                else:
                    confidence = (user_cos + item_cos)/2
                    confidence = tf.nn.sigmoid(confidence)

                mf_loss = tf.square(self.label - x_ij)
                mf_loss = confidence * self.mark * mf_loss + (1 - self.mark) * mf_loss

                unique_user_idx, _ = tf.unique(self.user_idx)
                unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)

                unique_item_idx, _ = tf.unique(self.item_idx)
                unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)

                l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))

                self.factual_loss = tf.reduce_mean(mf_loss) + self._lamb * l2_loss

            with tf.variable_scope("counter_factual_loss"):
                sampled_users = tf.nn.embedding_lookup(self.user_embeddings, self.sampled_user_idx)
                sampled_users_bias = tf.nn.embedding_lookup(self.user_bias_embeddings, self.sampled_user_idx)
                sampled_items = tf.nn.embedding_lookup(self.item_embeddings, self.sampled_item_idx)
                sampled_item_bias = tf.nn.embedding_lookup(self.item_bias_embeddings, self.sampled_item_idx)

                x_ij = tf.reduce_sum(tf.multiply(sampled_users, sampled_items),
                                     axis=1) + sampled_users_bias + sampled_item_bias

                unif_users = tf.nn.embedding_lookup(self.unif_user_embeddings, self.sampled_user_idx)
                unif_users_bias = tf.nn.embedding_lookup(self.unif_user_bias_embeddings, self.sampled_user_idx)
                unif_items = tf.nn.embedding_lookup(self.unif_item_embeddings, self.sampled_item_idx)
                unif_item_bias = tf.nn.embedding_lookup(self.unif_item_bias_embeddings, self.sampled_item_idx)

                unif_label = tf.reduce_sum(tf.multiply(unif_users, unif_items),
                                           axis=1) + unif_users_bias + unif_item_bias

                user_cos = tf.reduce_sum(tf.multiply(sampled_users, unif_users), axis=1) / (
                        tf.sqrt(tf.reduce_sum(tf.square(sampled_users), axis=1)) *
                        tf.sqrt(tf.reduce_sum(tf.square(unif_users), axis=1)))
                item_cos = tf.reduce_sum(tf.multiply(sampled_items, unif_items), axis=1) / (
                        tf.sqrt(tf.reduce_sum(tf.square(sampled_items), axis=1)) *
                        tf.sqrt(tf.reduce_sum(tf.square(unif_items), axis=1)))

                if self._measure == 'exp':
                    confidence = tf.exp((user_cos + item_cos) / 2)
                    confidence = tf.clip_by_value(confidence, 0, 1)
                else:
                    confidence = (user_cos + item_cos)/2
                    confidence = tf.nn.sigmoid(confidence)

                mf_loss = confidence * (tf.square(unif_label - x_ij))

                unique_user_idx, _ = tf.unique(self.sampled_user_idx)
                unique_users = tf.nn.embedding_lookup(self.user_embeddings, unique_user_idx)

                unique_item_idx, _ = tf.unique(self.sampled_item_idx)
                unique_items = tf.nn.embedding_lookup(self.item_embeddings, unique_item_idx)

                l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))
                self.cf_loss = tf.reduce_mean(mf_loss) + self._lamb * l2_loss

            with tf.variable_scope('loss'):
                self._loss = self.factual_loss + self._lamb2 * self.cf_loss

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

    def get_batches(self, user_item_pairs, marks, rating_matrix, batch_size):
        batches = []

        index_shuf = np.arange(len(user_item_pairs))
        np.random.shuffle(index_shuf)
        user_item_pairs = user_item_pairs[index_shuf]
        for i in range(int(len(user_item_pairs) / batch_size)):

            ui_pairs = user_item_pairs[i * batch_size: (i + 1) * batch_size, :]
            ui_pairs = ui_pairs.astype('int32')

            label = np.asarray(rating_matrix[ui_pairs[:, 0], ui_pairs[:, 1]])[0]

            mark = np.asarray(marks[ui_pairs[:, 0], ui_pairs[:, 1]])[0]

            sampled_user_idx = np.random.choice(self._num_users, batch_size)
            sampled_item_idx = np.random.choice(self._num_items, batch_size)

            batches.append([ui_pairs[:, 0], ui_pairs[:, 1], label, mark, sampled_user_idx, sampled_item_idx])

        return batches

    def train_model(self, matrix_train, marks, matrix_valid, epoch, metric='AUC', topK=50, is_topK=False):
        user_item_matrix = lil_matrix(matrix_train)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            batches = self.get_batches(user_item_pairs, marks, matrix_train, self._batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2],
                             self.mark: batches[step][3],
                             self.sampled_user_idx: batches[step][4],
                             self.sampled_item_idx: batches[step][5],
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


def bridgemf2(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
              iteration=100, lam=0.01, lam2=0.01, rank=200, batch_size=500, learning_rate=1e-3, optimizer="LAdam",
              seed=0, gpu_on=False, problem=None, scene='r', metric='AUC', topK=50, is_topK=False, measure='sigmoid',
              **unused):
    progress = WorkSplitter()

    progress.section("Bridge-MF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Bridge-MF: Load the variables trained on S_t")
    if scene == 'r':
        unif_RQ = np.load('latent/pretrain/' + problem + 'unif_U_MF_10_r.npy')
        unif_Y = np.load('latent/pretrain/' + problem + 'unif_V_MF_10_r.npy')
        unif_uBias = np.load('latent/pretrain/' + problem + 'unif_uB_MF_10_r.npy')
        unif_vBias = np.load('latent/pretrain/' + problem + 'unif_vB_MF_10_r.npy')
    elif scene == 'c':
        unif_RQ = np.load('latent/pretrain/' + problem + 'unif_U_MF_10_r.npy')
        unif_Y = np.load('latent/pretrain/' + problem + 'unif_V_MF_10_r.npy')
        unif_uBias = np.load('latent/pretrain/' + problem + 'unif_uB_MF_10_r.npy')
        unif_vBias = np.load('latent/pretrain/' + problem + 'unif_vB_MF_10_r.npy')
    else:
        unif_RQ = np.load('latent/pretrain/' + problem + 'unif_U_MF_10_t.npy')
        unif_Y = np.load('latent/pretrain/' + problem + 'unif_V_MF_10_t.npy')
        unif_uBias = np.load('latent/pretrain/' + problem + 'unif_uB_MF_10_t.npy')
        unif_vBias = np.load('latent/pretrain/' + problem + 'unif_vB_MF_10_t.npy')

    progress.section("Bridge-MF: Training")
    marks = csr_matrix(matrix_train.shape)
    marks[(matrix_train != 0).nonzero()] = 1

    matrix_input = matrix_train + matrix_utrain
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = BridgeMF(m, n, rank, batch_size, lamb=lam, lamb2=lam2, learning_rate=learning_rate, measure=measure,
                     optimizer=Regularizer[optimizer], gpu_on=gpu_on, init_unif_U=unif_RQ, init_unif_V=unif_Y,
                     init_unif_uBias=unif_uBias, init_unif_vBias=unif_vBias)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, marks, matrix_valid, iteration, metric, topK, is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias