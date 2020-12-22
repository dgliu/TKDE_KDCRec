import tensorflow as tf

Regularizer = {
    "Adam": tf.train.AdamOptimizer,
    "RMSProp": tf.train.RMSPropOptimizer,
    "SGD": tf.train.GradientDescentOptimizer,
    "LAdam": tf.contrib.opt.LazyAdamOptimizer,
    # "Momentum": tf.train.MomentumOptimizer,
}