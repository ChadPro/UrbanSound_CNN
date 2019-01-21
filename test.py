# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
from nets import cnn_net_convs

c1 = tf.ones([6,40,173,1])
y = cnn_net_convs.cnn_net(c1)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    r = sess.run(y)
    print r.shape
