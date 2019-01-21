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
def one_hot(labels):
    onehots = []
    for label in labels:
        letter = [0 for _ in range(10)]
        letter[label] = 1
        onehots.append(letter)
    onehots = np.array(onehots)
    return onehots

train_x = pickle.load(open('./train_x.dat', 'rb'))
train_y = pickle.load(open('./train_y.dat', 'rb'))
train_y = one_hot(train_y)
train_len = train_y.shape[0]

val_x = pickle.load(open('./val_x.dat', 'rb'))
val_y = pickle.load(open('./val_y.dat', 'rb'))
val_y = one_hot(val_y)
val_len = val_y.shape[0]

# test_x = pickle.load(open('./test_x.dat', 'rb'))
# test_y = pickle.load(open('./test_y.dat', 'rb'))
# test.len = test_y.shape[0]

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
with tf.name_scope("input"):
    x_input = tf.placeholder(tf.float32, [32,40,173,1], name="x-input")
    y_input = tf.placeholder(tf.float32, [32,10], name="y-input")
    x_val_input = tf.placeholder(tf.float32, [32,40,173,1], name="x-val-input")
    y_val_input = tf.placeholder(tf.float32, [32,10], name="y-val-input")
with tf.name_scope("Cnn_Net"):
    net = conv2d_block(x_input, [3,3,1,32], [1,1,1,1], scope="conv1")
    net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool1")
    net = conv2d_block(net, [3,3,32,32], [1,1,1,1], scope="conv2")
    net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool2")
    net = fc_block(net, 4096, None, activation=ACTIVATION, flatten=True, scope="fc1")
    net = fc_block(net, 4096, None, activation=ACTIVATION, scope="fc2")
with tf.name_scope("Forward_Propagation"):
    y = fc_block(net, 10, None, scope="output")
with tf.name_scope("Cnn_Net"):
    net = conv2d_block(x_val_input, [3,3,1,32], [1,1,1,1], scope="conv1")
    net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool1")
    net = conv2d_block(net, [3,3,32,32], [1,1,1,1], scope="conv2")
    net = maxpool_block(net, [1,2,2,1], [1,2,2,1], scope="pool2")
    net = fc_block(net, 4096, None, activation=ACTIVATION, flatten=True, scope="fc1")
    net = fc_block(net, 4096, None, activation=ACTIVATION, scope="fc2")
with tf.name_scope("Forward_Propagation"):
    y_val = fc_block(net, 10, None, scope="output")
global_step = tf.Variable(0, trainable=False)

with tf.name_scope("Calc_Loss"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean
    cross_entropy_val = tf.nn.softmax_cross_entropy_with_logits(labels=y_val_input, logits=y_val)
    cross_entropy_mean_val = tf.reduce_mean(cross_entropy_val)
    loss_val = cross_entropy_mean_val
with tf.name_scope("Back_Train"):
    learning_rate = tf.train.exponential_decay(0.01 ,global_step, 1000, 0.99)  
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step) 
with tf.name_scope("Calc_Acc"):
    correct_prediction = tf.equal(tf.argmax(y_input), tf.argmax(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    correct_prediction_val = tf.equal(tf.argmax(y_val_input), tf.argmax(y_val))
    accuracy_val = tf.reduce_mean(tf.cast(correct_prediction_val, tf.float32))

tf.summary.scalar("loss_train", loss)
tf.summary.scalar("loss_val", loss_val)
tf.summary.scalar("acc_train", accuracy)
tf.summary.scalar("acc_val", accuracy_val)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./log_dir/", tf.get_default_graph())
saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    for i in range(10000):
        data_indexs = np.random.choice(train_len, 32)
        inputs_x = train_x[data_indexs]
        inputs_x = np.expand_dims(inputs_x, axis=3)
        inputs_y = train_y[data_indexs]
        _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x_input:inputs_x, y_input:inputs_y})
        
        if i%100 == 0:
            val_index = np.random.choice(val_len, 32)
            inputs_x_val = val_x[val_index]
            inputs_x_val = np.expand_dims(inputs_x_val, axis=3)
            inputs_y_val = val_y[val_index]
            summary_str, loss_val_value, step_val = sess.run([merged, loss_val, global_step], feed_dict={x_val_input:inputs_x_val, y_val_input:inputs_y_val, x_input:inputs_x, y_input:inputs_y})
            writer.add_summary(summary_str, i)
            print "########### step : {0} ############".format(str(step_val))
            print "     loss      = {0}                ".format(str(loss_value))
            print "     loss_val  = {0}                  ".format(str(loss_val_value))

        if i%500 == 0:
            saver.save(sess, "./models/model.ckpt")        
    
    writer.close()

    
