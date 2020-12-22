import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.ae import ae
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix
from utils.io import sample_from_sc_with_mode, load_yaml, find_single_best_hyperparameters


class RefineAE(object):
    def __init__(self, input_dim, embed_dim, batch_size,
                 init_norm_X, init_norm_Y, init_norm_xBias, init_norm_yBias,
                 init_unif_X, init_unif_Y, init_unif_xBias, init_unif_yBias,
                 lamb=0.01,
                 confidence=0.9,
                 learning_rate=1e-4,
                 optimizer=tf.train.AdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._input_dim = self._output_dim = input_dim
        self._embed_dim = embed_dim
        self.norm_X = init_norm_X
        self.norm_Y = init_norm_Y
        self.norm_xBias = init_norm_xBias
        self.norm_yBias = init_norm_yBias
        self.unif_X = init_unif_X
        self.unif_Y = init_unif_Y
        self.unif_xBias = init_unif_xBias
        self.unif_yBias = init_unif_yBias
        self._lamb = lamb
        self._confidence = confidence
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('refine-ae'):
            # Placehoder
            self.inputs = tf.placeholder(tf.float32, (None, self._input_dim))

            # Import Pre-Trained Variables
            self.norm_encode_weights = tf.get_variable('norm_encode_weights', initializer=self.norm_X)
            self.norm_encode_bias = tf.get_variable('norm_encode_bias', initializer=self.norm_xBias)
            self.norm_decode_weights = tf.get_variable('norm_decode_weights', initializer=self.norm_Y)
            self.norm_decode_bias = tf.get_variable('norm_decode_bias', initializer=self.norm_yBias)

            self.unif_encode_weights = tf.get_variable('unif_encode_weights', initializer=self.unif_X,
                                                       trainable=False)
            self.unif_encode_bias = tf.get_variable('unif_encode_bias', initializer=self.unif_xBias,
                                                    trainable=False)
            self.unif_decode_weights = tf.get_variable('unif_decode_weights', initializer=self.unif_Y,
                                                       trainable=False)
            self.unif_decode_bias = tf.get_variable('unif_decode_bias', initializer=self.unif_yBias,
                                                    trainable=False)

            with tf.variable_scope('refine_label'):
                self.unif_encoded = tf.nn.relu(tf.matmul(self.inputs, self.unif_encode_weights) + self.unif_encode_bias)
                self.unif_prediction = tf.matmul(self.unif_encoded, self.unif_decode_weights) + self.unif_decode_bias

                mask = tf.where(tf.not_equal(self.inputs, 0), tf.ones(tf.shape(self.inputs)),
                                tf.zeros(tf.shape(self.inputs)))
                predict_label = self.unif_prediction * mask

                _predict_label = (predict_label - tf.reduce_min(predict_label, 1, keep_dims=True)) / (
                        tf.reduce_max(predict_label, 1, keep_dims=True) - tf.reduce_min(predict_label, 1, keep_dims=True))

                self.predict_label = tf.where(tf.is_nan(_predict_label), tf.zeros_like(_predict_label), _predict_label)
                self.refined_label = self.inputs + self._confidence * self.predict_label

            with tf.variable_scope('loss'):
                self.norm_encoded = tf.nn.relu(tf.matmul(self.inputs, self.norm_encode_weights) + self.norm_encode_bias)

                self.norm_prediction = tf.matmul(self.norm_encoded, self.norm_decode_weights) + self.norm_decode_bias

                l2_loss = tf.nn.l2_loss(self.norm_encode_weights) + tf.nn.l2_loss(self.norm_decode_weights)

                mf_loss = tf.square(self.refined_label - self.norm_prediction * mask)

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

    def train_model(self, matrix_train, matrix_valid, epoch=100, metric='AUC', topK=50, is_topK=False):
        batches = self.get_batches(matrix_train, self._batch_size)

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
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
            encoded = self.sess.run(self.norm_encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ), self.sess.run(self.norm_encode_weights), self.sess.run(self.norm_encode_bias)

    def get_Y(self):
        return self.sess.run(self.norm_decode_weights)

    def get_yBias(self):
        return self.sess.run(self.norm_decode_bias)


def refineae(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
             iteration=100, lam=0.01, confidence=0.9, alpha=0.8, rank=200, batch_size=256, learning_rate=1e-4,
             optimizer="Adam", seed=0, gpu_on=False, problem=None, scene='r', metric='AUC', topK=50, is_topK=False,
             mode='random', sample_num=2700, **unused):
    progress = WorkSplitter()

    progress.section("Refine-AE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Refine-AE: Sample a subset from S_c and construct new S_c and S_t")
    matrix_train, matrix_utrain = sample_from_sc_with_mode(matrix_train, matrix_utrain, sample_num,
                                                           pos_ratio=alpha, neg_ratio=1 - alpha, mode=mode)

    progress.section("Refine-MF: Pre-train M_c and M_t separately on the new data set")

    table_path = load_yaml('config/global.yml', key='path')['tables']
    df_norm = find_single_best_hyperparameters(table_path + problem + "ae_tuning_r.csv", 'AUC')
    RQ, X, xBias, Yt, yBias = ae(matrix_train, matrix_valid,
                                 matrix_utrain=matrix_utrain,
                                 iteration=df_norm['iter'],
                                 rank=df_norm['rank'],
                                 gpu_on=gpu_on,
                                 lam=df_norm['lambda'],
                                 seed=0,
                                 batch_size=df_norm['batch_size'],
                                 source=None)
    norm_X, norm_Y, norm_xBias, norm_yBias = X, Yt, xBias, yBias

    df_unif = find_single_best_hyperparameters(table_path + problem + "unif_ae_tuning_r.csv", 'AUC')
    RQ, X, xBias, Yt, yBias = ae(matrix_train, matrix_valid,
                                 matrix_utrain=matrix_utrain,
                                 iteration=df_unif['iter'],
                                 rank=df_unif['rank'],
                                 gpu_on=gpu_on,
                                 lam=df_unif['lambda'],
                                 seed=0,
                                 batch_size=df_unif['batch_size'],
                                 source='unif')
    unif_X, unif_Y, unif_xBias, unif_yBias = X, Yt, xBias, yBias

    progress.section("Refine-AE: Training")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = RefineAE(n, rank, batch_size, lamb=lam, confidence=confidence, learning_rate=learning_rate,
                     optimizer=Regularizer[optimizer], gpu_on=gpu_on, init_norm_X=norm_X, init_norm_Y=norm_Y,
                     init_norm_xBias=norm_xBias, init_norm_yBias=norm_yBias, init_unif_X=unif_X, init_unif_Y=unif_Y,
                     init_unif_xBias=unif_xBias, init_unif_yBias=unif_yBias)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_valid, iteration, metric, topK, is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias
