# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import pickle
import os
import glob
import numpy as np
import librosa
import sklearn

test_x = pickle.load(open('./test_x.dat', 'rb'))
test_y = pickle.load(open('./test_y.dat', 'rb'))

print type(test_x)
print type(test_y)

print test_x.shape
print test_y.shape