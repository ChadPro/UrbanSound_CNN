# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import tensorflow as tf

ACTIVATION = tf.nn.relu
STDDEV = 0.01

""" 0. Load Datasets
"""
test_x = pickle.load(open('./test_x.dat', 'rb'))
test_x = np.expand_dims(test_x, axis=3)
test_y = pickle.load(open('./test_y.dat', 'rb'))
test_len = test_y.shape[0]

""" 1. CNN Net Blocks
"""
def conv2d_block(inputs, dw_size, strides, downsample=False, is_training=True, padding="SAME", scope=""):
    _stride = strides
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        conv_deep = dw_size[-1]
        conv_weights = tf.get_variable("weights", dw_size, initializer=tf.truncated_normal_initializer(stddev=STDDEV))
        conv_biases = tf.get_variable("bias", conv_deep, initializer=tf.constant_initializer(0.))
        conv2d = tf.nn.conv2d(inputs, conv_weights, strides=_stride, padding=padding)
        net = ACTIVATION(tf.nn.bias_add(conv2d, conv_biases))
    return net

def maxpool_block(inputs, pool_size, strides, downsample=True, is_training=True, padding="SAME", scope=""):
    _stride = strides
    with tf.name_scope(scope):
        pool = tf.nn.max_pool(inputs, ksize=pool_size, strides=_stride, padding=padding)
    return pool

def fc_block(inputs, outputs, regularizer, activation=None, flatten=False, is_dropout=False, is_training=True, scope=""):
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
    return fc

def cnn_nets(inputs):
    with tf.name_scope("Cnn_Net"):
        net = conv2d_block(inputs, [3,3,1,32], [1,1,1,1], scope="conv1")
        net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool1")
        net = conv2d_block(net, [3,3,32,32], [1,1,1,1], scope="conv2")
        net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool2")
        net = fc_block(net, 4096, None, activation=ACTIVATION, flatten=True, scope="fc1")
        net = fc_block(net, 4096, None, activation=ACTIVATION, scope="fc2")
        net = fc_block(net, 10, None, scope="output")
    return net

inputs_x = tf.placeholder(tf.float32, [test_len,40,173,1], name="x-input")
inputs_y = tf.placeholder(tf.float32, [test_len,], name="y-input")

y = cnn_nets(inputs_x)
y = tf.nn.softmax(y)
saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver.restore(sess, "./models/model.ckpt")
    r, labels = sess.run([y, inputs_y], feed_dict={inputs_x:test_x, inputs_y:test_y})
    labels = labels.astype(np.int32)
    rr = np.argmax(r, axis=1)
    print rr[0:20]
    print "####"
    print labels[0:20]
