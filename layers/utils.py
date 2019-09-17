"""Utilities for DS-Softmax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def add_flags_hyper_parameters(flags):
    """Add hyper parameters to flags """
    flags.DEFINE_integer("nexperts", 2, "number of experts for gating")
    flags.DEFINE_float("importance_loss_coef", 30, "smoothing factor of gating function")
    flags.DEFINE_float("lasso_loss_coef", 0.003, "lasso loss for last layer")
    flags.DEFINE_float("expert_lasso_loss_coef", 0.003, "expert lasso loss for last layer")
    flags.DEFINE_bool("complex_gating", False, "Use complex gating function (two layers)")
    flags.DEFINE_bool("noise", False, "Use noise")
    flags.DEFINE_string("incremental_save", "", "the last incremental files")
    flags.DEFINE_integer("pruning_start", 0, "prune start epoch")
    flags.DEFINE_float("pruning_cutoff", 0.03, "prune columns with group lasso less than cutoff")
    flags.DEFINE_integer("partial_update_epoch", 1000, "partially update variables")
    flags.DEFINE_integer("async_update", -1, "async_update update expert and gate")
    return flags


def add_dimension_group_lasso(var, dim=0):
    """Returns a function that creates one layer

    Args:
        var: an variable
        dim: dimension for group lasso

    Returns:
        a tensor
    """
    var = tf.pow(var, 2)
    var = tf.reduce_mean(var, axis=dim) + 1e-8
    var = tf.pow(var, 1.0 / 2)
    return var


def linear_fn(in_features, out_features):
    """Returns a function that creates one layer

    Args:
        in_features: an integer
        out_features: an integer

    Returns:
        a unary function
    """

    def my_fn(x):
        w = tf.get_variable("softmax_weight", [in_features, out_features])
        y = tf.matmul(x, w)
        # need init, remove for pass graph building
        mask = tf.get_variable("softmax_mask", [out_features], trainable=False)
        y = tf.multiply(y, mask)
        return y

    return my_fn


def save_variables(session, save_path, expert_vars, mask_vars):
    """Save variables for softmax

    Args:
        session: tensorflow session
        save_path: save path for weights and masks
        expert_vars: softmax weight variables
        mask_vars: mask variables
    """
    print('Storing softmax variables')
    weight_variables_name = [var.name for var in expert_vars]
    mask_variables_name = [var.name for var in mask_vars]
    weights = np.stack(session.run(weight_variables_name))
    masks = np.stack(session.run(mask_variables_name))
    return np.savez(save_path, weights, masks)


def restore_variables(session, save_path, incremental_assign_ops):
    """Restore variables for softmax and clone softmax if stored softmax is less

    Args:
        session: tensorflow session
        save_path: save path for weights and masks
        incremental_assign_ops: all placeholders and assign ops
    """
    weight_placeholders, weight_assign_ops, mask_placeholders, mask_assign_ops = incremental_assign_ops
    tmp = np.load(save_path)
    weights, masks = tmp['arr_0'], tmp['arr_1']
    previous_nexperts = len(masks)
    copy = int(len(weight_assign_ops) * 1.0 / previous_nexperts)
    print('Loading stored softmax variables')
    for i in range(previous_nexperts):
        for j in range(copy):
            index = i * copy + j
            session.run(weight_assign_ops[index].name,
                        feed_dict={weight_placeholders[index].name: weights[i]})
            session.run(mask_assign_ops[index].name,
                        feed_dict={mask_placeholders[index].name: masks[i]})


def get_sparsity(session, mask_vars, gates):
    """get sparsity under current mask and gates
    """
    masks = session.run([var.name for var in mask_vars])
    sparsity = [np.mean(mask) for mask in masks]

    gates = np.argmax(gates, axis=1)
    bins = np.histogram(gates, bins=len(sparsity))[0]
    bins = bins * 1.0 / np.sum(bins)
    return 1 - np.dot(bins, sparsity)


def np_add_dimension_glasso(var, dim=0):
    """numpy version of group lasso
    """
    return np.power(np.mean(np.power(var, 2), axis=dim) + 1e-8, 1 / 2.)


def print_experts_info(session, expert_vars, cutoff):
    """get experts information 
    """
    experts = session.run([var.name for var in expert_vars])
    lasso = [np_add_dimension_glasso(var, 0) for var in experts]
    sparsity = [len(np.where(l < cutoff)[0]) for l in lasso]
    print(','.join(np.array(sparsity).astype(str)))
