# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import pickle
import numpy as np
import sklearn

train_x = pickle.load(open('./train_x.dat', 'rb'))
train_y = pickle.load(open('./train_y.dat', 'rb'))

val_x = pickle.load(open('./val_x.dat', 'rb'))
val_y = pickle.load(open('./val_y.dat', 'rb'))

print "########## Train Dataset ###############"
print "MFCC features shape : "
print train_x.shape
print "MFCC labels shape : "
print train_y.shape

print "########## Val Dataset #################"
print "MFCC features shape : "
print val_x.shape
print "MFCC labels shape : "
print val_y.shape
