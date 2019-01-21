# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import tensorflow as tf
from nets import cnn_net_simple
from nets import net_factory
from py_extend import vgg_acc

ACTIVATION = tf.nn.relu
STDDEV = 0.01

tf.app.flags.DEFINE_bool('regularizer', False, 'If use regularizer.')
tf.app.flags.DEFINE_bool('dropout', False, 'If use dropout.')
tf.app.flags.DEFINE_string('net', "cnn_net_simple", 'Chose net.')
FLAGS = tf.app.flags.FLAGS

def one_hot(labels):
    onehots = []
    for label in labels:
        letter = [0 for _ in range(10)]
        letter[label] = 1
        onehots.append(letter)
    onehots = np.array(onehots)
    return onehots

def train():
    train_x = pickle.load(open('./train_x.dat', 'rb'))
    train_y = pickle.load(open('./train_y.dat', 'rb'))
    train_y = one_hot(train_y)
    train_len = train_y.shape[0]

    val_x = pickle.load(open('./val_x.dat', 'rb'))
    val_y = pickle.load(open('./val_y.dat', 'rb'))
    val_y = one_hot(val_y)
    val_len = val_y.shape[0]

    net_cls = net_factory.get_network(FLAGS.net)

    with tf.name_scope("input"):
        x_input = tf.placeholder(tf.float32, [32,40,173,1], name="x-input")
        y_input = tf.placeholder(tf.float32, [32,10], name="y-input")
        x_val_input = tf.placeholder(tf.float32, [32,40,173,1], name="x-val-input")
        y_val_input = tf.placeholder(tf.float32, [32,10], name="y-val-input")
        train_flag = tf.placeholder(tf.bool, name="train-flag")

    with tf.name_scope("Forward_Propagation"):
        if FLAGS.regularizer:
            regularizer = tf.contrib.layers.l2_regularizer(0.001, scope="regularizer")
        else:
            regularizer = None
        y = net_cls.cnn_net(x_input, is_training=train_flag, regularizer=regularizer, is_dropout=FLAGS.dropout)
        y_val = net_cls.cnn_net(x_val_input)
    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope("Calc_Loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if FLAGS.regularizer:
            loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        else:
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

        for i in range(1000000):
            data_indexs = np.random.choice(train_len, 32)
            inputs_x = train_x[data_indexs]
            inputs_x = np.expand_dims(inputs_x, axis=3)
            inputs_y = train_y[data_indexs]
            _, loss_value, step = sess.run([train_step, loss, global_step], feed_dict={x_input:inputs_x, y_input:inputs_y, train_flag:True})
        
            if i%100 == 0:
                val_index = np.random.choice(val_len, 32)
                inputs_x_val = val_x[val_index]
                inputs_x_val = np.expand_dims(inputs_x_val, axis=3)
                inputs_y_val = val_y[val_index]
                summary_str, loss_val_value, step_val = sess.run([merged, loss_val, global_step], feed_dict={x_val_input:inputs_x_val, y_val_input:inputs_y_val, x_input:inputs_x, y_input:inputs_y, train_flag:True})
                writer.add_summary(summary_str, i)
                print "########### step : {0} ############".format(str(step_val))
                print "     loss      = {0}                ".format(str(loss_value))
                print "     loss_val  = {0}                  ".format(str(loss_val_value))

            if i%500 == 0:
                saver.save(sess, "./models/model.ckpt")        
        writer.close()

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()