# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

from nets import cnn_net_simple
from nets import cnn_net_deep
from nets import cnn_net_convs


net_map = {
    'cnn_net_simple' : cnn_net_simple,
    'cnn_net_deep' : cnn_net_deep,
    'cnn_net_convs' : cnn_net_convs
    }


def get_network(name):
    if name not in net_map:
        raise ValueError('Name of net unknown %s' % name)
    return net_map[name]