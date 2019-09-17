from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from . import expert_utils
from . import utils


class Softmax(object):
    def __init__(self,
                 in_dimension,
                 out_dimension,
                 num_experts,
                 importance_weight=0,
                 lasso_weight=0,
                 expert_lasso_weight=0,
                 pruning_threshold=0):

        """Create a Softmax.

        Args:
            in_dimension: dimension of input
            out_dimension: dimension of output (number of classes)
            num_experts: number of experts in mixture of experts
            importance_weight: weight for balancing term
            lasso_weight: weight for lasso loss on output class level
            expert_lasso_weight: weight for loss on expert level
            pruning_threshold: class with lasso smaller than this will be pruned

        Returns:
            a Softmax object
        """
        self.in_dimension = in_dimension
        self.out_dimension = out_dimension

        # hyper-parameter for ds-softmax
        self.num_experts = num_experts
        self.importance_weight = importance_weight
        self.lasso_weight = lasso_weight
        self.expert_lasso_weight = expert_lasso_weight
        self.pruning_threshold = pruning_threshold

        # init pruning op
        self.prune_op = None

        # inti variables list
        self.expert_vars = None
        self.mask_vars = None
        self.gate_vars = None

    def forward(self, input_, train=True):
        """Build DS-Softmax

        Args:
            input_: input
            train: train mode

        Returns:
            output: output activation
            gates: gating activation
            loss_tuple: tuple of three losses: importance, lasso, expert_lasso
        """
        linear_fn = utils.linear_fn(self.in_dimension, self.out_dimension)
        output, importance_loss, gates = expert_utils.local_moe(input_, train, linear_fn, self.num_experts,
                                                                loss_coef=self.importance_loss_coef)

        self.init_vars()
        self.init_prune_op()

        lasso_loss, expert_lasso_loss = self.get_lasso_losses()
        losses_tuple = (importance_loss, lasso_loss, expert_lasso_loss)
        return output, gates, losses_tuple

    def get_train_ops(self, loss, optimizer):
        """Returns train ops for training the expert only, gate only and others

        Args:
            loss: loss
            optimizer: optimizer

        Returns:
            train_expert: train softmax weight only
            train_gate: train gating network only
            train_other: train other components rather than softmax and gating network
        """
        expert_grad = tf.gradients(loss, self.expert_vars)
        gate_grad = tf.gradients(loss, self.gate_vars)
        other_vars = [var for var in tf.trainable_variables() if 'softmax' not in var.name and 'moe' not in var.name]
        other_grad = tf.gradients(loss, other_vars)

        train_expert = optimizer.apply_gradients(zip(expert_grad, self.expert_vars), name='train_expert')
        train_gate = optimizer.apply_gradients(zip(gate_grad, self.gate_vars), name='train_gate')
        try:
            train_other = optimizer.apply_gradients(zip(other_grad, other_vars), name='train_other')
        except:
            train_other = None
        return train_expert, train_gate, train_other

    def init_vars(self):
        """Init three list of variables"""
        self.expert_vars = [var for var in tf.trainable_variables() if 'softmax_weight' in var.name]
        self.mask_vars = [var for var in tf.global_variables() if 'softmax_mask' in var.name]
        self.gate_vars = [var for var in tf.global_variables() if 'top_k_gating' in var.name]

    def init_prune_op(self):
        """Init prune op
        """
        prune_ops = []
        lasso_losses = [utils.add_dimension_group_lasso(var, 0) for var in self.expert_vars]
        masks = [tf.less(lasso, tf.constant(self.pruning_cutoff)) for lasso in lasso_losses]

        for i in range(len(lasso_losses)):
            mask = masks[i]
            assign = self.mask_vars[i].assign(
                tf.where(mask, tf.zeros_like(self.mask_vars[i]), tf.ones_like(self.mask_vars[i])))
            prune_ops.append(assign)

            mask = tf.tile(tf.expand_dims(mask, 0), [self.expert_vars[i].shape[0], 1])
            assign = self.expert_vars[i].assign(tf.where(mask, tf.zeros_like(self.expert_vars[i]), self.expert_vars[i]))
            prune_ops.append(assign)

        self.prune_op = tf.group(prune_ops)

    def get_lasso_losses(self):
        """Returns sum of lasso loss on class level and expert level

        Args:
            expert_vars: variables of experts

        Returns:
            weighted_lasso_loss: weighted lasso loss on class level
            weighted_expert_lasso_loss: weighted lasso loss on expert level
        """

        lasso_loss = utils.add_dimension_group_lasso(tf.stack(self.expert_vars), 1)
        expert_lasso_loss = utils.add_dimension_group_lasso(lasso_loss, 0)
        lasso_loss = tf.reduce_sum(lasso_loss)
        expert_lasso_loss = tf.reduce_sum(expert_lasso_loss)

        weighted_lasso_loss = self.lasso_weight * lasso_loss
        weighted_expert_lasso_loss = self.expert_lasso_weight * expert_lasso_loss
        return weighted_lasso_loss, weighted_expert_lasso_loss
