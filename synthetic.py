from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import layers

flags = tf.flags

# synthetic data hyper-parameter
flags.DEFINE_integer('nsuperclass', 100, 'number of superclass')
flags.DEFINE_integer('nitem', 100, 'number of item in each subclass')
flags.DEFINE_integer('dimension', 100, 'dimension of input data')
flags.DEFINE_float('distance', 100, 'distance between superclass')
flags.DEFINE_float('ratio', 10, 'distance ratio between superclass and subclass')

# training hyper-parameter
flags.DEFINE_integer('epoch', 100, 'epoch to train')
flags.DEFINE_integer('batch_size', 500, 'batch size')
flags.DEFINE_float('learning_rate', 0.003, 'learning rate')
flags.DEFINE_float('keep_prob', 0.8, 'dropout')

# moe softmax hyper-parameter
flags = layers.util.add_flags_hyper_parameters(flags)
FLAGS = flags.FLAGS


def generate_data(nsuperclass, nsubclass, nitem, dimension, distance, ratio):
    """Generate synthetic data

    Args:
        nsuperclass: number of superclass
        nsubclass: number of subclass
        nitem: number of item in each class
        dimension: number of dimension for each item
        distance: the variance of gaussian to generate data
        ratio: the variance ratio between superclass and subclass

    Returns:
        input_: the input data
        output: the corresponding subclass
        superoutput: the corresponding superclass
    """
    cov = np.zeros((dimension, dimension))
    np.fill_diagonal(cov, distance)
    superclasses = np.random.multivariate_normal(np.zeros(dimension), cov, nsuperclass)

    np.fill_diagonal(cov, distance * 1.0 / ratio)
    classes = [np.random.multivariate_normal(superclass, cov, nsubclass) for superclass in superclasses]

    classes = np.concatenate(classes, axis=0)
    np.fill_diagonal(cov, distance * 1.0 / ratio / ratio)
    input_ = [np.random.multivariate_normal(c, cov, nitem) for c in classes]
    input_ = np.concatenate(input_)
    output = np.repeat(np.arange(nsuperclass * nsubclass), nitem)
    superoutput = np.repeat(np.arange(nsuperclass), (nsubclass * nitem))
    return input_, output, superoutput


class Model(object):
    """The two-layer model."""

    def __init__(self, is_training=True):
        self.input_ = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.dimension))
        self.target = tf.placeholder(tf.int32, shape=FLAGS.batch_size)

        if is_training:
            inputs_ = tf.nn.dropout(self.input_, FLAGS.keep_prob)

        hidden = tf.contrib.layers.fully_connected(inputs_, FLAGS.dimension)
        softmax = layers.softmax.Softmax(FLAGS.dimension, FLAGS.nsuperclass * FLAGS.nsuperclass, FLAGS.nexperts,
                                         importance_weight=FLAGS.importance_loss_coef,
                                         lasso_weight=FLAGS.lasso_loss_coef,
                                         expert_lasso_weight=FLAGS.expert_lasso_loss_coef,
                                         pruning_threshold=FLAGS.pruning_cutoff)

        logits, gates, (importance_loss, lasso_loss, expert_lasso_loss) = softmax.forward(hidden, train=is_training)
        self.gates = gates
        self.softmax = softmax
        self.importance_loss = importance_loss
        self.lasso_loss = lasso_loss
        self.expert_lasso_loss = expert_lasso_loss

        # accuracy
        self.logits = logits
        self.prediction = prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        self.accuracy = tf.contrib.metrics.accuracy(prediction, self.target)

        # cross entropy loss
        targets = tf.one_hot(self.target, FLAGS.nsuperclass * FLAGS.nsuperclass)
        self.cost = tf.losses.softmax_cross_entropy(targets, logits)

        self.loss = final_loss = self.cost + self.importance_loss + self.lasso_loss + self.expert_lasso_loss

        self.prune_op = softmax.prune_op
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        self.train_expert, self.train_gate, self.train_other = softmax.get_train_ops(final_loss, optimizer)


def run_epoch(session, model, input_, target, eval_op=None):
    """Runs the model on the given data."""
    costs = 0.0
    fetches = {
        "accuracy": model.accuracy,
        "cost": model.cost,
        "importance_loss": model.importance_loss.name,
        "lasso_loss": model.lasso_loss.name,
        "expert_lasso_loss": model.expert_lasso_loss.name,
    }

    if eval_op is not None:
        for i in range(len(eval_op)):
            fetches["eval_op_%d" % i] = eval_op[i]

    num_items = len(input_)
    indices = np.arange(num_items)
    np.random.shuffle(indices)

    for i in range(int(num_items / FLAGS.batch_size)):
        mindices = indices[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
        minput_, mtarget = input_[mindices], target[mindices]

        feed_dict = {
            model.input_: minput_,
            model.target: mtarget,
        }

        vals = session.run(fetches, feed_dict)

        cost = vals["cost"]
        accuracy = vals["accuracy"]
        importance_loss = vals["importance_loss"]
        lasso_loss = vals["lasso_loss"]
        expert_lasso_loss = vals["expert_lasso_loss"]

        costs += cost
        if i % 100 == 0:
            print("Iter %d: accuracy %.3f cost: %.3f import: %.3f lasso: %.3f elasso: %.3f" %
                  (i + 1, accuracy, cost, importance_loss, lasso_loss, expert_lasso_loss))


def main(_):
    input_, target, superoutput = generate_data(FLAGS.nsuperclass, FLAGS.nsuperclass, FLAGS.nitem,
                                                FLAGS.dimension, FLAGS.distance, FLAGS.ratio)
    model = Model()

    sv = tf.train.Supervisor()
    config_proto = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with sv.managed_session(config=config_proto) as session:
        for i in range(FLAGS.epoch):
            print("Epoch: %d" % (i + 1))
            run_epoch(session, model, input_, target, eval_op=[model.train_expert, model.train_gate, model.train_other])
            layers.util.print_experts_info(session, model.softmax.expert_vars, FLAGS.pruning_cutoff)
            if i > FLAGS.pruning_start:
                session.run(model.softmax.prune_op)
                masks = session.run([var for var in model.softmax.mask_vars])
                if i == FLAGS.epoch - 1:
                    np.save('masks.npy', np.array(masks))

                masks = [np.mean(mask) for mask in masks]
                print('masks')
                print(','.join(np.array(masks).astype(str)))


if __name__ == "__main__":
    tf.app.run()
