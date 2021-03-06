import numpy as np
from tqdm import tqdm
import tensorflow as tf
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix


class BridgeAE(object):
    def __init__(self, input_dim, embed_dim, batch_size,
                 init_norm_X, init_norm_Y, init_norm_xBias, init_norm_yBias,
                 init_unif_X, init_unif_Y, init_unif_xBias, init_unif_yBias,
                 lamb=0.01,
                 lamb2=0.01,
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
        self._lamb2 = lamb2
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('bridge-ae'):
            # Placehoder
            self.inputs = tf.placeholder(tf.float32, (None, self._input_dim))
            self.sample_idx = tf.placeholder(tf.float32, (None, self._input_dim))

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

            with tf.variable_scope('factual_loss'):
                self.encoded = tf.nn.relu(tf.matmul(self.inputs, self.norm_encode_weights) + self.norm_encode_bias)
                self.prediction = tf.matmul(self.encoded, self.norm_decode_weights) + self.norm_decode_bias

                l2_loss = tf.nn.l2_loss(self.norm_encode_weights) + tf.nn.l2_loss(self.norm_decode_weights)

                mask = tf.where(tf.not_equal(self.inputs, 0), tf.ones(tf.shape(self.inputs)),
                                tf.zeros(tf.shape(self.inputs)))
                mf_loss = tf.square(self.inputs - self.prediction * mask)

                self.factual_loss = tf.reduce_mean(mf_loss) + self._lamb * tf.reduce_mean(l2_loss)

            with tf.variable_scope('counter_factual_loss'):
                self.unif_encoded = tf.nn.relu(tf.matmul(self.inputs, self.unif_encode_weights) + self.unif_encode_bias)
                self.unif_prediction = tf.matmul(self.unif_encoded, self.unif_decode_weights) + self.unif_decode_bias

                self.cf_loss = tf.reduce_mean(tf.square((self.prediction - self.unif_prediction) * self.sample_idx))

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
    def get_batches(rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        sample_idx = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                temp = rating_matrix[batch_index * batch_size:]
                batches.append(temp)
                sample_idx.append(np.random.randint(2, size=np.shape(temp)))
            else:
                temp = rating_matrix[batch_index * batch_size:(batch_index + 1) * batch_size]
                batches.append(temp)
                sample_idx.append(np.random.randint(2, size=np.shape(temp)))
            batch_index += 1
            remaining_size -= batch_size
        return batches, sample_idx

    def train_model(self, matrix_train, matrix_valid, epoch=100, metric='AUC', topK=50, is_topK=False):
        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        for i in tqdm(range(epoch)):
            batches, sample_idx = self.get_batches(matrix_train, self._batch_size)
            for step in range(len(batches)):
                feed_dict = {self.inputs: batches[step].todense(),
                             self.sample_idx: sample_idx[step]}
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
        batches, _ = self.get_batches(rating_matrix, self._batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.inputs: batches[step].todense()}
            encoded = self.sess.run(self.encoded, feed_dict=feed_dict)
            RQ.append(encoded)

        return np.vstack(RQ), self.sess.run(self.norm_encode_weights), self.sess.run(self.norm_encode_bias)

    def get_Y(self):
        return self.sess.run(self.norm_decode_weights)

    def get_yBias(self):
        return self.sess.run(self.norm_decode_bias)


def oldbridgeae(matrix_train, matrix_valid, embeded_matrix=np.empty(0), matrix_utrain=None,
                iteration=100, lam=0.01, lam2=0.01, rank=200, batch_size=500, learning_rate=1e-4, optimizer="Adam",
                seed=0, gpu_on=False, problem=None, scene='r', metric='AUC', topK=50, is_topK=False, **unused):
    progress = WorkSplitter()

    progress.section("Old-Bridge-AE: Set the random seed")
    np.random.seed(seed)
    tf.set_random_seed(seed)

    progress.section("Old-Bridge-AE: Load the variables trained on S_c/S_t")
    if scene == 'r':
        norm_X = np.load('latent/pretrain/' + problem + 'X_AE_200_r.npy')
        norm_Y = np.load('latent/pretrain/' + problem + 'Y_AE_200_r.npy')
        norm_xBias = np.load('latent/pretrain/' + problem + 'xB_AE_200_r.npy')
        norm_yBias = np.load('latent/pretrain/' + problem + 'yB_AE_200_r.npy')

        unif_X = np.load('latent/pretrain/' + problem + 'unif_X_AE_200_r.npy')
        unif_Y = np.load('latent/pretrain/' + problem + 'unif_Y_AE_200_r.npy')
        unif_xBias = np.load('latent/pretrain/' + problem + 'unif_xB_AE_200_r.npy')
        unif_yBias = np.load('latent/pretrain/' + problem + 'unif_yB_AE_200_r.npy')
    elif scene == 'c':
        norm_X = np.load('latent/pretrain/' + problem + 'X_AE_200_c.npy')
        norm_Y = np.load('latent/pretrain/' + problem + 'Y_AE_200_c.npy')
        norm_xBias = np.load('latent/pretrain/' + problem + 'xB_AE_200_c.npy')
        norm_yBias = np.load('latent/pretrain/' + problem + 'yB_AE_200_c.npy')

        unif_X = np.load('latent/pretrain/' + problem + 'unif_X_AE_200_r.npy')
        unif_Y = np.load('latent/pretrain/' + problem + 'unif_Y_AE_200_r.npy')
        unif_xBias = np.load('latent/pretrain/' + problem + 'unif_xB_AE_200_r.npy')
        unif_yBias = np.load('latent/pretrain/' + problem + 'unif_yB_AE_200_r.npy')
    else:
        norm_X = np.load('latent/pretrain/' + problem + 'X_AE_200_r.npy')
        norm_Y = np.load('latent/pretrain/' + problem + 'Y_AE_200_r.npy')
        norm_xBias = np.load('latent/pretrain/' + problem + 'xB_AE_200_r.npy')
        norm_yBias = np.load('latent/pretrain/' + problem + 'yB_AE_200_r.npy')

        unif_X = np.load('latent/pretrain/' + problem + 'unif_X_AE_200_t.npy')
        unif_Y = np.load('latent/pretrain/' + problem + 'unif_Y_AE_200_t.npy')
        unif_xBias = np.load('latent/pretrain/' + problem + 'unif_xB_AE_200_t.npy')
        unif_yBias = np.load('latent/pretrain/' + problem + 'unif_yB_AE_200_t.npy')

    progress.section("Old-Bridge-AE: Training")
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = BridgeAE(n, rank, batch_size, lamb=lam, lamb2=lam2, learning_rate=learning_rate,
                     optimizer=Regularizer[optimizer], gpu_on=gpu_on, init_norm_X=norm_X, init_norm_Y=norm_Y,
                     init_norm_xBias=norm_xBias, init_norm_yBias=norm_yBias, init_unif_X=unif_X, init_unif_Y=unif_Y,
                     init_unif_xBias=unif_xBias, init_unif_yBias=unif_yBias)

    RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_valid, iteration, metric, topK, is_topK)

    model.sess.close()
    tf.reset_default_graph()

    return RQ, X, xBias, Y, yBias