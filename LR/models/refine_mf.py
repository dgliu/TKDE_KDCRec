import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.mf import mf
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix
from utils.io import sample_from_sc_with_mode, load_yaml, find_single_best_hyperparameters


class RefineMF(object):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 init_norm_U, init_norm_V, init_norm_uBias, init_norm_vBias,
                 init_unif_U, init_unif_V, init_unif_uBias, init_unif_vBias,
                 lamb=0.01,
                 confidence=0.9,
                 learning_rate=1e-3,
                 optimizer=tf.contrib.opt.LazyAdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._num_users = num_users
        self._num_items = num_items
        self._embed_dim = embed_dim
        self.norm_U = init_norm_U
        self.norm_V = init_norm_V
        self.norm_uBias = init_norm_uBias
        self.norm_vBias = init_norm_vBias
        self.unif_U = init_unif_U
        self.unif_V = init_unif_V
        self.unif_uBias = init_unif_uBias
        self.unif_vBias = init_unif_vBias
        self._lamb = lamb
        self._confidence = confidence
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('refine-mf'):
            # Placehoder
            self.user_idx = tf.placeholder(tf.int32, [None])
            self.item_idx = tf.placeholder(tf.int32, [None])
            self.label = tf.placeholder(tf.float32, [None])

            # Import Pre-Trained Variables
            self.unif_user_embeddings = tf.get_variable(name='unif_users', initializer=self.unif_U,
                                                        trainable=False)
            self.unif_item_embeddings = tf.get_variable(name='unif_items', initializer=self.unif_V,
                                                        trainable=False)
            self.unif_user_bias_embeddings = tf.get_variable(name='unif_users_bias', initializer=self.unif_uBias,
                                                             trainable=False)
            self.unif_item_bias_embeddings = tf.get_variable(name='unif_items_bias', initializer=self.unif_vBias,
                                                             trainable=False)

            self.norm_user_embeddings = tf.get_variable(name='norm_users', initializer=self.norm_U)
            self.norm_item_embeddings = tf.get_variable(name='norm_items', initializer=self.norm_V)
            self.norm_user_bias_embeddings = tf.get_variable(name='norm_users_bias', initializer=self.norm_uBias)
            self.norm_item_bias_embeddings = tf.get_variable(name='norm_items_bias', initializer=self.norm_vBias)

        with tf.variable_scope("refine_label"):
            unif_users = tf.nn.embedding_lookup(self.unif_user_embeddings, self.user_idx)
            unif_users_bias = tf.nn.embedding_lookup(self.unif_user_bias_embeddings, self.user_idx)
            unif_items = tf.nn.embedding_lookup(self.unif_item_embeddings, self.item_idx)
            unif_item_bias = tf.nn.embedding_lookup(self.unif_item_bias_embeddings, self.item_idx)

            predict_label = tf.reduce_sum(
                tf.multiply(unif_users, unif_items), axis=1) + unif_users_bias + unif_item_bias
            _predict_label = (predict_label - tf.reduce_min(predict_label)) / (
                    tf.reduce_max(predict_label) - tf.reduce_min(predict_label))

            self.refined_label = self.label + self._confidence * _predict_label

        with tf.variable_scope("mf_loss"):
            users = tf.nn.embedding_lookup(self.norm_user_embeddings, self.user_idx)
            users_bias = tf.nn.embedding_lookup(self.norm_user_bias_embeddings, self.user_idx)
            items = tf.nn.embedding_lookup(self.norm_item_embeddings, self.item_idx)
            item_bias = tf.nn.embedding_lookup(self.norm_item_bias_embeddings, self.item_idx)

            x_ij = tf.reduce_sum(tf.multiply(users, items), axis=1) + users_bias + item_bias
            mf_loss = tf.reduce_mean(tf.square(self.refined_label - x_ij))

        with tf.variable_scope('l2_loss'):
            unique_user_idx, _ = tf.unique(self.user_idx)
            unique_users = tf.nn.embedding_lookup(self.norm_user_embeddings, unique_user_idx)

            unique_item_idx, _ = tf.unique(self.item_idx)
            unique_items = tf.nn.embedding_lookup(self.norm_item_embeddings, unique_item_idx)

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

    def get_batches(self, user_item_pairs, matrix_train, batch_size):
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

    def train_model(self, matrix_train, matrix_valid, epoch=100, metric='AUC', topK=50, is_topK=False):
        user_item_matrix = lil_matrix(matrix_train)
        user_item_pairs = np.asarray(user_item_matrix.nonzero()).T

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            batches = self.get_batches(user_item_pairs, matrix_train, self._batch_size)
            for step in range(len(batches)):
                feed_dict = {self.user_idx: batches[step][0],
                             self.item_idx: batches[step][1],
                             self.label: batches[step][2]
                             }
                training = self.sess.run([self._train], feed_dict=feed_dict)

            RQ, Y, xBias, yBias = self.sess.run([self.norm_user_embeddings,
                                                 self.norm_item_embeddings,
                                                 self.norm_user_bias_embeddings,
                                                 self.norm_item_bias_embeddings])
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


def refinemf(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
             iteration=100, lam=0.01, confidence=0.9, alpha=0.8, rank=200, batch_size=500, learning_rate=1e-3,
             optimizer="LAdam", seed=0, gpu_on=False, problem=None, scene='r', metric='AUC', topK=50, is_topK=False,
             mode='random', sample_num=2700, **unused):
    progress = WorkSplitter()

    progress.section("Refine-MF: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Refine-MF: Sample a subset from S_c and construct new S_c and S_t")
    matrix_train, matrix_utrain = sample_from_sc_with_mode(matrix_train, matrix_utrain, sample_num,
                                                           pos_ratio=alpha, neg_ratio=1-alpha, mode=mode)

    progress.section("Refine-MF: Pre-train M_c and M_t separately on the new data set")

    table_path = load_yaml('config/global.yml', key='path')['tables']
    df_norm = find_single_best_hyperparameters(table_path + problem + "mf_tuning_r.csv", 'AUC')
    RQ, X, xBias, Yt, yBias = mf(matrix_train, matrix_valid,
                                 matrix_utrain=matrix_utrain,
                                 iteration=df_norm['iter'],
                                 rank=df_norm['rank'],
                                 gpu_on=gpu_on,
                                 lam=df_norm['lambda'],
                                 seed=0,
                                 batch_size=df_norm['batch_size'],
                                 source=None)
    norm_RQ, norm_Y, norm_uBias, norm_vBias = RQ, Yt.T, xBias, yBias

    df_unif = find_single_best_hyperparameters(table_path + problem + "unif_mf_tuning_r.csv", 'AUC')
    RQ, X, xBias, Yt, yBias = mf(matrix_train, matrix_valid,
                                 matrix_utrain=matrix_utrain,
                                 iteration=df_unif['iter'],
                                 rank=df_unif['rank'],
                                 gpu_on=gpu_on,
                                 lam=df_unif['lambda'],
                                 seed=0,
                                 batch_size=df_unif['batch_size'],
                                 source='unif')
    unif_RQ, unif_Y, unif_uBias, unif_vBias = RQ, Yt.T, xBias, yBias

    progress.section("Refine-MF: Training")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = RefineMF(m, n, rank, batch_size, lamb=lam, confidence=confidence, learning_rate=learning_rate,
                     optimizer=Regularizer[optimizer], gpu_on=gpu_on, init_norm_U=norm_RQ, init_norm_V=norm_Y,
                     init_norm_uBias=norm_uBias, init_norm_vBias=norm_vBias, init_unif_U=unif_RQ, init_unif_V=unif_Y,
                     init_unif_uBias=unif_uBias, init_unif_vBias=unif_vBias)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_valid, iteration, metric, topK, is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias