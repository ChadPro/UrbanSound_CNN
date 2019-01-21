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

parent_dir = './UrbanSound8K/audio/'
train_dir = 'train/'
val_dir = 'val/'
test_dir = 'fold10/'
file_name = '*.wav'

train_files = glob.glob(os.path.join(parent_dir, train_dir, file_name))
val_files = glob.glob(os.path.join(parent_dir, val_dir, file_name))
test_files = glob.glob(os.path.join(parent_dir, test_dir, file_name))

def audios_check(filenames):
    l_audios = []

    for filepath in filenames:
        print filepath
        x, sr = librosa.load(filepath)
        l_audios.append(len(x))
    np_audios = np.array(l_audios)
    len_max = np.max(np_audios)
    len_min = np.min(np_audios)
    return len_min, len_max

def load_clip(filename):
    x, sr = librosa.load(filename)
    if 88200 - x.shape[0] > 0:
        x = np.pad(x, (0,88200-x.shape[0]), 'constant')
    else:
        x = x[0:88200]
    return x, sr

def extract_feature(filename):
    x, sr = load_clip(filename)
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    norm_mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
    return norm_mfccs

def load_dataset(filenames):
    features, labels = np.empty((0,40,173)), np.empty(0)
    cnt = 0
    cnt_all = len(filenames)
    
    for filename in filenames:
        mfccs = extract_feature(filename)
        features = np.append(features,mfccs[None],axis=0)
        cnt+=1
        if(cnt%100==0):
            print([str(cnt)+' / '+str(cnt_all)+' finished'])
        labels = np.append(labels, filename.split('/')[-1].split('-')[1])
    return np.array(features), np.array(labels, dtype=np.int)


""" 0. check audio files
"""
# len_min, len_max = audios_check(train_files)
# print "######### Train 音频文件统计 ############"
# print "    最长 " + str(len_max)  #88375
# print "    最短 " + str(len_min)  #1103
# len_min, len_max = audios_check(val_files)
# print "######### Val 音频文件统计 ############"
# print "    最长 " + str(len_max)  #89009
# print "    最短 " + str(len_min)    #3749
# len_min, len_max = audios_check(test_files)
# print "######### Test 音频文件统计 ############"
# print "    最长 " + str(len_max)  #88200
# print "    最短 " + str(len_min)  #5264

""" 1. extract features and save
"""
# train_x, train_y = load_dataset(train_files)
# pickle.dump(train_x, open('./train_x.dat', 'wb'))
# pickle.dump(train_y, open('./train_y.dat', 'wb'))
# val_x, val_y = load_dataset(val_files)
# pickle.dump(val_x, open('./val_x.dat', 'wb'))
# pickle.dump(val_y, open('./val_y.dat', 'wb'))
# test_x, test_y = load_dataset(test_files)
# pickle.dump(test_x, open('./test_x.dat', 'wb'))
# pickle.dump(test_y, open('./test_y.dat', 'wb'))