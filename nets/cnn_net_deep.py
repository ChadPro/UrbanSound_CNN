# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import tensorflow as tf

ACTIVATION = tf.nn.relu
DEFAULT_OUTPUT_NODE = 10
STDDEV = 0.01

def conv2d_block(inputs, dw_size, strides, regularizer, downsample=False, is_training=True, padding="SAME", scope=""):
    _stride = strides
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv_deep = dw_size[-1]
        conv_weights = tf.get_variable("weights", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        conv_biases = tf.get_variable("bias", conv_deep, initializer=tf.constant_initializer(0.))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(conv_weights))
            tf.add_to_collection("losses", regularizer(conv_biases))
        conv2d = tf.nn.conv2d(inputs, conv_weights, strides=_stride, padding=padding)
        net = ACTIVATION(tf.nn.bias_add(conv2d, conv_biases))
    return net

def maxpool_block(inputs, pool_size, strides, downsample=True, is_training=True, padding="SAME", scope=""):
    _stride = strides
    with tf.name_scope(scope):
        pool = tf.nn.max_pool(inputs, ksize=pool_size, strides=_stride, padding=padding)
    return pool

def fc_block(inputs, outputs, regularizer, activation=None, flatten=False, is_dropout=False, is_training=None, scope=""):
    if flatten:
        net_shape = inputs.get_shape()
        nodes = tf.math.multiply(tf.math.multiply(net_shape[1], net_shape[2]), net_shape[3])
        reshaped = tf.reshape(inputs, [net_shape[0], nodes])
        inputs = reshaped
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        fc_weights = tf.get_variable("weights", [inputs.get_shape()[1], outputs], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        if regularizer != None:
            tf.add_to_collection("losses", regularizer(fc_weights))
        fc_biases = tf.get_variable("bias", [outputs], initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        fc = tf.nn.bias_add(tf.matmul(inputs, fc_weights), fc_biases)
        if activation:
            fc = activation(fc)
        if is_dropout:
            fc = tf.cond(is_training, lambda: tf.nn.dropout(fc, 0.5), lambda: fc)
            # fc = tf.nn.dropout(fc, 0.5)
    return fc

def cnn_net(inputs, \
                num_classes=DEFAULT_OUTPUT_NODE, \
                is_training=None, \
                reuse=None, \
                white_bal=False, \
                regularizer=None, \
                is_dropout=False, \
                scope='cnn_net_simple'):
    
    with tf.name_scope(scope):
        net = conv2d_block(inputs, [3,3,1,32], [1,1,1,1], regularizer, scope="conv1")
        net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool1")
        net = conv2d_block(net, [3,3,32,32], [1,1,1,1], regularizer, scope="conv2")
        net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool2")   # (batch, 10,44,32)
        net = conv2d_block(net, [3,3,32,64], [1,1,1,1], regularizer, scope="conv3")
        net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool3")   # (batch, 5,22,64)
        net = conv2d_block(net, [3,3,64,64], [1,1,1,1], regularizer, scope="conv4")
        net = conv2d_block(net, [3,3,64,64], [1,1,1,1], regularizer, scope="conv5")
        net = conv2d_block(net, [3,3,64,32], [1,1,1,1], regularizer, scope="conv6")
        net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool4")
        net = fc_block(net, 4096, regularizer, activation=ACTIVATION, flatten=True, is_dropout=is_dropout, is_training=is_training, scope="fc1")
        net = fc_block(net, 4096, regularizer, activation=ACTIVATION, is_dropout=is_dropout, is_training=is_training, scope="fc2")
        net = fc_block(net, 10, regularizer, is_training=is_training, scope="output")

    return net